import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

VALID_STATUSES = {"pending", "confirmed"}


def _extract_purpose_from_docstring(docstring: str) -> str:
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in ("Args:", "Returns:", "Raises:", "Note:", "Example:", "Internally")):
            continue
        return stripped.strip()
    return ""


@dataclass
class FunctionEntry:

    name: str
    signature: str
    docstring: str
    purpose: str = ""
    code_body: str = ""
    status: str = "pending"
    source: str = "additional"

    def to_prompt_block(self) -> str:

        docstring = self.docstring.strip()

        if docstring and not docstring.startswith('"""'):
            docstring = f'"""\n{docstring}\n"""'

        return f"{docstring}\n{self.signature}".strip()

    def to_api_info(self) -> Dict[str, str]:

        implementation = self.code_body

        if implementation.lstrip().startswith("def "):
            lines = implementation.splitlines()

            implementation = "\n".join(lines[1:])
        return {
            "method_name": self.name,
            "docstring": self.docstring,
            "purpose": self.purpose or _extract_purpose_from_docstring(self.docstring),
            "signature": self.signature,
            "implementation": implementation,
        }


class LocalRAG:

    def __init__(self, registry_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent
        functions_dir = base_dir / "functions"

        self.registry_path = Path(
            registry_path or functions_dir / "local_rag_registry.json"
        )
        self._addition_path = Path(functions_dir / "addition.py")
        self._primitive_path = Path(functions_dir / "primitive.py")

        self.entries: Dict[str, FunctionEntry] = {}

        self.load()
        self._sync_primitives()

    @property
    def addition_path(self):
        return self._addition_path

    def load(self) -> None:
        if not self.registry_path.exists():
            self.entries = {}
            return

        with self.registry_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        self.entries = {
            item["name"]: FunctionEntry(**item)
            for item in data.get("functions", [])
            if item.get("name")
        }

    def save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "functions": [
                asdict(entry)
                for entry in sorted(self.entries.values(), key=lambda item: item.name)
                if entry.source != "primitive"
            ]
        }

        with self.registry_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def _sync_primitives(self) -> None:
        for entry in self._scan_file(
            self._primitive_path, status="confirmed", source="primitive"
        ):
            if entry.name not in self.entries:
                self.upsert(entry, persist=False)

    def upsert(
        self,
        entry: FunctionEntry,
        persist: bool = True,
        write_addition: bool = False,
    ) -> FunctionEntry:
        if entry.status not in VALID_STATUSES:
            raise ValueError(f"Invalid LocalRAG status: {entry.status}")

        self.entries[entry.name] = entry

        if write_addition and entry.source != "primitive":
            self.write_additional_functions()

        if persist:
            self.save()

        return entry

    def register_function(
        self,
        name: str,
        signature: str,
        docstring: str,
        code_body: str,
        purpose: str = "",
        status: str = "pending",
        source: str = "additional",
        write_addition: bool = True,
    ) -> FunctionEntry:
        entry = FunctionEntry(
            name=name,
            signature=signature.strip(),
            docstring=docstring.strip(),
            code_body=code_body.strip("\n"),
            purpose=purpose.strip() if purpose else _extract_purpose_from_docstring(docstring),
            status=status,
            source=source,
        )
        return self.upsert(entry, write_addition=write_addition)

    def promote(self, name: str) -> None:
        if name not in self.entries:
            raise KeyError(f"Unknown LocalRAG function: {name}")
        self.entries[name].status = "confirmed"
        self.save()

    def remove(self, name: str) -> None:
        if name in self.entries:
            del self.entries[name]
            self.write_additional_functions()
            self.save()

    def retrieve(
        self,
        include_pending: bool = False,
        exclude_primitives: bool = False,
    ) -> List[FunctionEntry]:
        allowed_statuses = VALID_STATUSES if include_pending else {"confirmed"}
        return [
            entry
            for entry in sorted(self.entries.values(), key=lambda e: e.name)
            if entry.status in allowed_statuses
            and not (exclude_primitives and entry.source == "primitive")
        ]

    def prompt_context(
        self,
        include_pending: bool = False,
        exclude_primitives: bool = False,
    ) -> str:
        return "\n\n".join(
            entry.to_prompt_block()
            for entry in self.retrieve(
                include_pending=include_pending,
                exclude_primitives=exclude_primitives,
            )
        )

    def api_entries(
        self,
        include_pending: bool = False,
        exclude_primitives: bool = False,
    ) -> List[Dict[str, str]]:
        return [
            entry.to_api_info()
            for entry in self.retrieve(
                include_pending=include_pending,
                exclude_primitives=exclude_primitives,
            )
        ]

    def write_additional_functions(self) -> None:
        additional_entries = [
            entry
            for entry in self.entries.values()
            if entry.source != "primitive" and entry.code_body.strip()
        ]

        lines = [
            '"""Dynamically generated VADAR additional functions."""',
            "",
            "import math",
            "from agents.functions.primitive import *",
            "",
        ]

        for entry in sorted(additional_entries, key=lambda item: item.name):
            code = entry.code_body.strip()
            if not code.startswith("def "):
                code = f"{entry.signature}\n{code}"
            lines.append(code)
            lines.append("")  

        self._addition_path.write_text(
            "\n".join(lines).rstrip() + "\n", encoding="utf-8"
        )

    def _scan_file(
        self, file_path: Path, status: str, source: str
    ) -> Iterable[FunctionEntry]:

        if not file_path.exists():
            return []
        source_text = file_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source_text)
        except SyntaxError:
            return []

        entries = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if source == "primitive" and not self._has_register_primitive_decorator(
                node
            ):
                continue

            raw_code_body = ast.get_source_segment(source_text, node) or ""
            signature = self._extract_signature(raw_code_body)
            docstring = ast.get_docstring(node) or "TODO: add a function description."

            code_body = self._strip_docstring(raw_code_body, docstring)

            entries.append(
                FunctionEntry(
                    name=node.name,
                    signature=signature,
                    docstring=docstring,
                    code_body=code_body,
                    purpose=_extract_purpose_from_docstring(docstring),
                    status=status,
                    source=source,
                )
            )
        return entries

    def _has_register_primitive_decorator(self, node: ast.FunctionDef) -> bool:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "register_primitive":
                return True
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "register_primitive":
                    return True
        return False

    def _extract_signature(self, code_body: str) -> str:
        lines = code_body.splitlines()
        sig_lines = []
        in_signature = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def "):
                in_signature = True
            if in_signature:
                sig_lines.append(stripped)
                if stripped.endswith(":") and not stripped.startswith('"""'):
                    break
        if sig_lines:
            return " ".join(sig_lines)
        
        return code_body.strip().splitlines()[0] if code_body.strip() else ""

    @staticmethod
    def _strip_docstring(code_body: str, docstring: str) -> str:

        if not docstring or docstring.startswith("TODO"):
            return code_body
        lines = code_body.splitlines()
        result = []
        in_docstring = False
        skip_blank_after = False
        for line in lines:
            stripped = line.strip()

            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True

                if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                    in_docstring = False
                    skip_blank_after = True
                continue
            if in_docstring:

                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                    skip_blank_after = True
                continue
            if skip_blank_after and stripped == "":
                skip_blank_after = False
                continue
            skip_blank_after = False
            result.append(line)
        return "\n".join(result)