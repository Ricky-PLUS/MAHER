import os
import sys
import json
import base64
import io
import re
import textwrap
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from PIL import Image
import signal
import linecache
import runpy
import shutil
import traceback

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from engine.engine_utils import (
    Generator,
    replace_tabs_with_spaces,
    TimeoutException,
    timeout_handler,
)
from prompts.signature_prompt import SIGNATURE_PROMPT

from prompts.api_prompt import API_PROMPT


def _extract_purpose_from_docstring(docstring: str) -> str:
    """Extract a one-sentence purpose from the first meaningful line of a docstring."""
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in ("Args:", "Returns:", "Raises:", "Note:", "Example:", "Internally")):
            continue
        return stripped.strip()
    return ""


class Agent:
    def __init__(
        self,
        model_name="deepseek-reasoner", #deepseek-reasoner	deepseek-v4-pro
        write_results=True,
        api_key_path="./api.key",
        dataset="rgpt",
    ):
        self.generator = Generator(model_name, api_key_path=api_key_path)
        self.write_results = write_results
        self.dataset = dataset
        self._results_folder = None

    @staticmethod
    def _write_error_log(results_folder, entry):
        if not results_folder:
            return
        log_path = os.path.join(results_folder, "error_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, indent=2, ensure_ascii=False) + "\n---\n")


class SignatureAgent(Agent):

    def __init__(
        self,
        predef_signatures,
        model_name="deepseek-reasoner",
        write_results=True,
        headers=[],
        local_rag=None,
        dataset="omni3d",
        critic=None,
    ):
        super().__init__(model_name, write_results, dataset=dataset)
        self.signatures = predef_signatures
        self.predef_signatures = predef_signatures
        self.headers = headers
        self.local_rag = local_rag
        self.critic = critic or CriticAgent(model_name=model_name)
        self.generated_docstrings = []
        self.generated_signatures = []
        self.generated_purposes = []
        self.generated_headers = []
        self.method_names = []
        self.review = 5

        for header in headers:
            self.generated_docstrings.append(header["docstring"])
            self.generated_signatures.append(header["signature"])
            self.generated_purposes.append(header.get("purpose", ""))
            self.generated_headers.append(header["docstring"] + header["signature"])

    def remove_substring(self, output, substring):

        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def _normalize_generated_signature(self, signature):
        match = re.compile(r"def (\w+)\s*\(").search(signature)
        if not match:
            return signature, None
        method_name = match.group(1)
        if method_name.startswith("_"):
            return signature, method_name
        normalized_name = f"_{method_name}"
        normalized_signature = re.sub(
            rf"\bdef\s+{re.escape(method_name)}\s*\(",
            f"def {normalized_name}(",
            signature,
            count=1,
        )
        return normalized_signature, normalized_name

    def _rag_context(self, query, include_pending=False):
        if not self.local_rag:
            return ""
        return self.local_rag.prompt_context(
            include_pending=include_pending,
            exclude_primitives=True,
        )

    def _primitive_context(self):
        if not self.local_rag:
            return ""
        return "\n\n".join(
            entry.to_prompt_block()
            for entry in self.local_rag.entries.values()
            if entry.source == "primitive"
        )

    def __call__(
        self,
        questions,
        prompt,
        ground_truth=None,
        critic_report=None,
        include_pending_tools=False,
    ):
        question_text = "\n\n".join(questions)
        rag_tools = self._rag_context(question_text, include_pending=include_pending_tools)
        primitives = self._primitive_context()
        signatures = self.signatures
        
        # import pdb; pdb.set_trace()
        output, _ = self.generator.generate(
            prompt.format(
                primitives=primitives,
                signatures=signatures,
                rag_tools=rag_tools,
                question=question_text,
                ground_truth=ground_truth or "",
                critic_report=json.dumps(critic_report, ensure_ascii=False)
                if critic_report
                else "",
            )
        )
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")

        docstrings = re.findall(r"<docstring>(.*?)</docstring>", output, re.DOTALL)
        signatures = re.findall(r"<signature>(.*?)</signature>", output, re.DOTALL)
        purposes = re.findall(r"<purpose>(.*?)</purpose>", output, re.DOTALL)
        normalized = [self._normalize_generated_signature(sig) for sig in signatures]
        signatures = [item[0] for item in normalized]

        self.generated_docstrings += docstrings
        self.generated_signatures += signatures
        self.generated_purposes += purposes
        headers = [doc + sig for doc, sig in zip(docstrings, signatures)]
        method_names = [
            normalized_name
            or re.compile(r"def\s+(\w+)\s*\(").search(sig).group(1)
            for sig, normalized_name in normalized
        ]
        
        rag_names = set(self.local_rag.entries.keys()) if self.local_rag else set()
        filtered_headers, filtered_method_names = [], []
        filtered_docstrings, filtered_signatures, filtered_purposes = [], [], []
        if len(purposes) < len(docstrings):
            purposes += [""] * (len(docstrings) - len(purposes))
        for doc, sig, hdr, name, pur in zip(docstrings, signatures, headers, method_names, purposes[:len(docstrings)]):
            if name in self.method_names or name in rag_names:
                continue
            filtered_docstrings.append(doc)
            filtered_signatures.append(sig)
            filtered_purposes.append(pur if pur else _extract_purpose_from_docstring(doc))
            filtered_headers.append(hdr)
            filtered_method_names.append(name)
        docstrings, signatures, purposes, headers, method_names = (
            filtered_docstrings, filtered_signatures, filtered_purposes, filtered_headers, filtered_method_names
        )
        if not method_names:
            return headers, output

        primitives = self._primitive_context()
        review_histories: Dict[int, list] = {}
        for idx in range(len(docstrings)):
            review_histories.setdefault(idx, [])
            for attempt in range(self.review):
                review = self.critic.review_signature(
                    primitive_context=primitives,
                    docstring=docstrings[idx],
                    signature=signatures[idx],
                    purpose=purposes[idx],
                    review_history=review_histories[idx],
                )
                if review.get("is_correct", True):
                    break
                if not review.get("suggestion", "").strip():
                    break
                review_histories[idx].append({
                    "issues": review.get("issues", ""),
                    "suggestion": review.get("suggestion", ""),
                })
                self._write_error_log(
                    self._results_folder,
                    {
                        "stage": "signature_review",
                        "function_name": method_names[idx],
                        "docstring": docstrings[idx],
                        "signature": signatures[idx],
                        "purpose": purposes[idx],
                        "review_issues": review.get("issues", ""),
                        "review_suggestion": review.get("suggestion", ""),
                        "review_attempt": len(review_histories[idx]),
                        "review_history": [
                            {"issues": e["issues"], "suggestion": e["suggestion"]}
                            for e in review_histories[idx][:-1]
                        ],
                    },
                )

                locked_name = method_names[idx]
                history_hint = ""
                if len(review_histories[idx]) > 1:
                    prev = "\n".join(
                        f"  Attempt {i}: issues={e['issues']}, suggestion={e['suggestion']}"
                        for i, e in enumerate(review_histories[idx][:-1], 1)
                    )
                    history_hint = (
                        f"\n===== PREVIOUS REVIEWS =====\n{prev}\n"
                        f"If previous suggestions contradict, fix the DOCSTRING/PURPOSE "
                        f"rather than flip-flopping the signature.\n"
                    )
                fix_prompt = (
                    f"===== PRIMITIVE REFERENCE =====\n{primitives}\n\n"
                    f"You previously generated this function:\n"
                    f"<purpose>\n{purposes[idx]}\n</purpose>\n"
                    f"<docstring>\n{docstrings[idx]}\n</docstring>\n"
                    f"<signature>\n{signatures[idx]}\n</signature>\n\n"
                    f"===== REVIEW FEEDBACK =====\n"
                    f"Issues: {review.get('issues', '')}\n"
                    f"Suggestion: {review.get('suggestion', '')}"
                    f"{history_hint}\n\n"
                    f"Revise the function to fix the issues. Prefer correcting the "
                    f"purpose and docstring — only change the signature (parameters / "
                    f"return type) if they are truly wrong. "
                    f"The function name MUST remain exactly `{locked_name}`.\n"
                    f"Output exactly one <purpose>...</purpose>, one <docstring>...</docstring>, "
                    f"followed by one <signature>...</signature>."
                )
                output, _ = self.generator.generate(fix_prompt)
                output = self.remove_substring(output, "```python")
                output = self.remove_substring(output, "```")
                new_docs = re.findall(r"<docstring>(.*?)</docstring>", output, re.DOTALL)
                new_sigs = re.findall(r"<signature>(.*?)</signature>", output, re.DOTALL)
                new_purs = re.findall(r"<purpose>(.*?)</purpose>", output, re.DOTALL)
                if new_docs and new_sigs:
                    revised_sig = re.sub(
                        r"\bdef\s+\w+\s*\(",
                        f"def {locked_name}(",
                        new_sigs[0],
                        count=1,
                    )
                    docstrings[idx] = new_docs[0]
                    signatures[idx] = revised_sig
                    if new_purs:
                        purposes[idx] = new_purs[0]

        self.method_names += method_names
        self.generated_headers += headers
        self.signatures += "\n\n".join(headers)
        self.headers += [
            {"method_name": method_name, "docstring": doc, "signature": sig, "purpose": pur}
            for doc, sig, method_name, pur in zip(docstrings, signatures, method_names, purposes[:len(method_names)])
        ]


        return headers, output

    def signatures_info(self):
        return [
            {"docstring": doc, "signature": sig}
            for doc, sig in zip(self.generated_docstrings, self.generated_signatures)
        ]

    def get_signatures(
        self,
        questions_data,
        images_folder_path,
        results_folder_path,
        prompt=None,
        question_batch_size=10,
        include_pending_tools=False,
        critic_report=None,
    ):
        prompt = SIGNATURE_PROMPT

        folder_name = "signature_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)
        self._results_folder = results_folder_path
        
        question_batches = []
        for i in range(0, len(questions_data), question_batch_size):
            question_batches.append(questions_data[i : i + question_batch_size])
        

        for question_batch in tqdm(question_batches):

            questions = [question_data["question"] for question_data in question_batch]
            ground_truth = [
                question_data.get("answer", "")
                for question_data in question_batch
            ]
            rag_tools = self._rag_context(
                "\n\n".join(questions),
                include_pending=include_pending_tools,
            )
            prompt_text = prompt.format(
                primitives=self._primitive_context(),
                signatures=self.signatures,
                rag_tools=rag_tools,
                question="\n\n".join(questions),
                ground_truth="\n".join([str(item) for item in ground_truth]),
                critic_report=json.dumps(critic_report, ensure_ascii=False)
                if critic_report
                else "",
            )

            headers, output = self(
                questions,
                prompt,
                ground_truth=ground_truth,
                critic_report=critic_report,
                include_pending_tools=include_pending_tools,
            )

            for question_data in question_batch:

                html_path = os.path.join(
                    results_folder_path,
                    f"image_{question_data['image_index']}_question_{question_data['question_index']}.html",
                )

                if self.write_results:
                    with open(html_path, "wb+") as file:

                        image = Image.open(
                            os.path.join(
                                images_folder_path, question_data["image_filename"]
                            )
                        )
                        image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                        rgb_image = image.convert("RGB")
                        image_io = io.BytesIO()
                        rgb_image.save(image_io, format="PNG")
                        image_bytes = base64.b64encode(image_io.getvalue()).decode(
                            "ascii"
                        )

                        file.write(
                            (f"<h1>{question_data['question']}</h1>\n").encode("utf-8")
                        )
                        file.write(
                            (
                                f"<img src='data:image/jpeg;base64,{image_bytes}'>\n"
                            ).encode("utf-8")
                        )

                        file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                        file.write(
                            (
                                f"<code>{prompt_text}</code>\n".replace("\n", "<br>")
                            ).encode("utf-8")
                        )

                        file.write((f"<h1>LLM Output</h1>\n").encode("utf-8"))
                        file.write(
                            (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                                "utf-8"
                            )
                        )

                        new_signatures = "\n\n".join(headers)
                        file.write((f"<h1>New Signatures</h1>\n").encode("utf-8"))
                        file.write(
                            (
                                f"<code>{new_signatures}</code>\n".replace("\n", "<br>")
                            ).encode("utf-8")
                        )

                        file.close()
            signatures_path = os.path.join(results_folder_path, "signatures.json")

            signatures_info = self.signatures_info()

            with open(signatures_path, "w+") as file:
                json.dump(signatures_info, file, indent=2, ensure_ascii=False)

        return signatures_path, signatures_info


class FuncAgent(Agent):

    def __init__(
        self,
        signature_agent,
        dataset,
        model_name="deepseek-reasoner",
        write_results=True,
        api=[],
        local_rag=None,
        critic=None,
    ):
        super().__init__(model_name, write_results)
        self.signature_agent = signature_agent
        self.dataset = dataset
        self.implementations = []
        self.api = api
        self.local_rag = local_rag or getattr(signature_agent, "local_rag", None)
        self.critic = critic or CriticAgent(model_name=model_name)
        self.error_counts = [0 for _ in range(len(self.signature_agent.method_names))]
        self.namespace = {}
        self.namespace_line = sys.maxsize
        self.trace_file_path = ""
        self.implemented = [
            False for _ in range(len(self.signature_agent.method_names))
        ]
        self.method_stack = []
        self.max_num_errors = 5
        self.pbar = tqdm(total=len(self.signature_agent.method_names))

    def _sync_signature_state(self):
        missing = len(self.signature_agent.method_names) - len(self.error_counts)
        if missing <= 0:
            return
        self.error_counts.extend([0 for _ in range(missing)])
        self.implemented.extend([False for _ in range(missing)])

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def __call__(
        self,
        method_name,
        docstring,
        signature,
        results_folder_path,
        prompt=None,
        api_info=None,
        critic_report=None,
    ):

        if prompt is None:
            prompt = API_PROMPT

        self._sync_signature_state()
        if self.implemented[self.signature_agent.method_names.index(method_name)]:
            return [
                api_info
                for api_info in self.api
                if api_info.get("method_name") == method_name
            ][0]["implementation"], ""

        
        self.pbar.set_description(
            f"Implementing {method_name} at error count {self.error_counts[self.signature_agent.method_names.index(method_name)]}"
        )

        generated_signatures = [
            header["docstring"] + header["signature"]
            for header in self.signature_agent.headers
            if header["method_name"] != method_name and
            self.error_counts[self.signature_agent.method_names.index(header["method_name"])] < self.max_num_errors
        ]
        generated_signatures = "\n\n".join(generated_signatures)
        rag_tools = ""
        if self.local_rag:
            rag_tools = self.local_rag.prompt_context(
                include_pending=True,
                exclude_primitives=True,
            )

        if api_info:
            messages = api_info["messages"] if api_info["messages"] is not None else None
            if critic_report and messages is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "A critic reported that this function has an implementation "
                            f"problem. Use this feedback when rewriting it:\n{json.dumps(critic_report, ensure_ascii=False)}"
                        ),
                    }
                )
                
            output, messages = self.generator.generate(
                "",
                messages=messages
            )
        else:
            messages = None
            primitives = self.signature_agent._primitive_context()
            output, messages = self.generator.generate(
                prompt.format(
                    primitives=primitives,
                    predef_signatures=self.signature_agent.predef_signatures,
                    generated_signatures=generated_signatures,
                    rag_tools=rag_tools,
                    docstring=docstring,
                    signature=signature,
                    critic_report=json.dumps(critic_report, ensure_ascii=False)
                    if critic_report
                    else "",
                ),
                messages=messages
            )

        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")

        implementation = re.findall(
            r"<implementation>(.*?)</implementation>", output, re.DOTALL
        )
        implementation = implementation[0]

        lines = implementation.split("\n")
        signature_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                signature_index = i
                lines[i] = signature
                break

        if signature_index is not None:
            implementation = "\n".join(lines[signature_index + 1 :])
        else:
            implementation = textwrap.indent(textwrap.dedent(implementation), "    ")
            
        purpose = ""
        for h in self.signature_agent.headers:
            if h.get("method_name") == method_name:
                purpose = h.get("purpose", "")
                break
        api_info = {
            "docstring": docstring,
            "signature": signature,
            "implementation": replace_tabs_with_spaces(implementation),
            "method_name": method_name,
            "purpose": purpose,
            "messages": messages,
            "error": None,
            "review_history": api_info.get("review_history", [])
            if api_info
            else [],
        }

        api_info = self.test_implementation(api_info, results_folder_path, prompt=prompt)
        
        
        if api_info["error"]:
            self._write_error_log(
                results_folder_path,
                {
                    "stage": "implementation_test",
                    "function_name": method_name,
                    "docstring": docstring,
                    "signature": signature,
                    "implementation": api_info.get("implementation", ""),
                    "error": api_info["error"],
                },
            )
            if (
                self.error_counts[self.signature_agent.method_names.index(method_name)]
                < self.max_num_errors
            ):
                api_info["messages"].append({"role": "user", "content": f"The implementation failed with error:\n{api_info['error']}\nPlease fix ONLY the function body logic. Keep the function name `{method_name}` and its signature exactly as defined — do NOT rename, reorder, or change parameters or return type. Output the corrected implementation in <implementation></implementation> tags."})
                self.method_stack.append(method_name)
                self(
                    method_name,
                    docstring,
                    signature,
                    results_folder_path,
                    prompt=prompt,
                    api_info=api_info
                )
        else:
            review_history = api_info.setdefault("review_history", [])
            review_result = self.critic(
                primitive_context=self.signature_agent._primitive_context(),
                docstring=api_info["docstring"],
                signature=api_info["signature"],
                implementation=api_info["implementation"],
                review_history=review_history,
            )
            if not review_result.get("is_correct", True):
                review_entry = {
                    "issues": review_result.get("issues", ""),
                    "suggestion": review_result.get("suggestion", ""),
                }
                review_history.append(review_entry)
                self._write_error_log(
                    results_folder_path,
                    {
                        "stage": "implementation_review",
                        "function_name": method_name,
                        "docstring": docstring,
                        "signature": signature,
                        "implementation": api_info.get("implementation", ""),
                        "review_issues": review_entry["issues"],
                        "review_suggestion": review_entry["suggestion"],
                        "review_attempt": len(review_history),
                        "review_history": [
                            {"issues": e["issues"], "suggestion": e["suggestion"]}
                            for e in review_history[:-1]
                        ],
                    },
                )
                issues = review_entry["issues"]
                suggestion = review_entry["suggestion"]
                if suggestion.strip():
                    idx = self.signature_agent.method_names.index(method_name)
                    self.error_counts[idx] += 1
                    self.implemented[idx] = False
                    if self.error_counts[idx] < self.max_num_errors:
                        api_info["messages"].append(
                            {
                                "role": "user",
                                "content": (
                                    f"Review found an error in the implementation:\n{issues}\n\n"
                                    f"Fix: {suggestion}\n\n"
                                    f"Fix ONLY the function body. Keep the name `{method_name}` "
                                    f"and signature parameters unchanged."
                                ),
                            }
                        )
                        self.method_stack.append(method_name)
                        self(
                            method_name,
                            docstring,
                            signature,
                            results_folder_path,
                            prompt=prompt,
                            api_info=api_info,
                        )
                    return "", ""

            self.api.append(api_info)
            self.implementations.append(implementation)
            if self.local_rag:
                rag_body = textwrap.indent(textwrap.dedent(implementation), "    ")
                self.local_rag.register_function(
                    name=method_name,
                    signature=signature,
                    docstring=docstring,
                    code_body=signature + "\n" + rag_body,
                    purpose=api_info.get("purpose", ""),
                    status="pending",
                    source="additional",
                    write_addition=True,
                )
            return implementation, output
        return "", ""

    def test_implementation(self, api_info, results_folder_path, prompt=API_PROMPT):
        method_name = (
            re.compile(r"def\s+(\w+)\s*\(").search(api_info["signature"]).group(1)
        )
        if self.error_counts[self.signature_agent.method_names.index(method_name)] >= 5:
            api_info["error"] = "Implementation failed"
            return api_info
        
        predef_api = []

        for signature_text in self.signature_agent.predef_signatures.split('\n\n"""'):
            if not signature_text.strip():
                continue  
            signature_start = signature_text.find("def ")
            if signature_start < 0:
                continue

            docstring = signature_text[:signature_start].strip()
            signature = signature_text[signature_start:].strip()

            _, returns = self._get_docstring_types(docstring, signature)

            implementation = self._get_return_code(returns)

            predef_api.append(
                {
                    "docstring": docstring,
                    "signature": signature,
                    "implementation": implementation,
                }
            )

        implementation_results_path = os.path.join(
            results_folder_path, f"{method_name}"
        )
        if os.path.exists(implementation_results_path):
            shutil.rmtree(implementation_results_path)
        os.makedirs(implementation_results_path)
        exec_env_path = os.path.join(implementation_results_path, "exec_env/")
        os.makedirs(exec_env_path)

        self.trace_file_path = os.path.join(exec_env_path, "trace.html")
        program_executable_path = os.path.join(exec_env_path, "executable_program.py")
        result_file = os.path.join(exec_env_path, "result.json")

        with open(program_executable_path, "w") as f:
            f.write("from typing import List, Dict, Tuple, Set, Optional, Union, Any\n")
            f.write("import math\n\n")

        try:
            from agents.functions.primitive import PRIMITIVES_REGISTRY
        except ImportError:
            PRIMITIVES_REGISTRY = {}

        primitive_names = set(PRIMITIVES_REGISTRY.keys()) if PRIMITIVES_REGISTRY else set()
        for method_info in predef_api:
            method_name_match = re.compile(r"def (\w+)\s*\(").search(method_info["signature"])
            func_name = method_name_match.group(1) if method_name_match else ""
            if func_name in primitive_names:
                continue 
            with open(program_executable_path, "a+") as f:
                f.write(method_info["signature"] + "\n")
                f.write(method_info["implementation"] + "\n\n")

        if self.local_rag:
            api_func_names = set()
            for info in self.api:
                match = re.compile(r"def (\w+)\s*\(").search(info.get("signature", ""))
                if match:
                    api_func_names.add(match.group(1))
            for name, entry in self.local_rag.entries.items():
                if entry.source == "primitive":
                    continue
                if name in api_func_names:
                    continue 
                if not entry.code_body.strip():
                    continue
                code = entry.code_body.strip()
                if not code.startswith("def "):
                    code = f"{entry.signature}\n{code}"
                with open(program_executable_path, "a+") as f:
                    f.write(code + "\n\n")

        arg_types, returns = self._get_docstring_types(api_info["docstring"], api_info.get("signature", ""))

        self.namespace = {}

        self.namespace.update(PRIMITIVES_REGISTRY)
        from typing import List, Dict, Tuple, Set, Optional, Union, Any
        self.namespace.update({
            "List": List, "Dict": Dict, "Tuple": Tuple,
            "Set": Set, "Optional": Optional, "Union": Union, "Any": Any,
        })

        _DEFAULT_TEST_IMAGE_PATH = ""
        
        for arg, type in arg_types:
            if type == "image":
                self.namespace.update({arg: _DEFAULT_TEST_IMAGE_PATH})
            elif arg == "image_path":
                self.namespace.update({arg: _DEFAULT_TEST_IMAGE_PATH})
            elif type == "int":
                self.namespace.update({arg: 25})
            elif type == "string":
                self.namespace.update({arg: ""})
            elif type == "float":
                self.namespace.update({arg: 1.0})
            elif type == "list":
                if "point" in arg.lower():
                    self.namespace.update({arg: [25, 25]})
                else:
                    self.namespace.update({arg: [25, 25, 50, 50]})
            elif type == "tuple":
                self.namespace.update({arg: (50, 50)})
            else:
                self.namespace.update({arg: 1})

        with open(program_executable_path, "a+") as file:
            for method_info in self.api:
                file.write(method_info["signature"] + "\n")
                file.write(method_info["implementation"] + "\n\n")

            file.write(api_info["signature"] + "\n")
            impl_lines = api_info["implementation"].split("\n")
            for line in impl_lines:
                file.write(line + "\n")
            file.write("\n\n# PROGRAM STARTS HERE\n")

            call_arg_names = [arg for arg, _ in arg_types]
            call_str = ", ".join(call_arg_names)
            file.write(f"final_result = {method_name}({call_str})\n")
            write_namespace_code = f"""
# WRITE NAMESPACE
import json
def is_serializable(obj):
    try:
        json.dumps(obj)
    except (TypeError, OverflowError):
        return False
    return True

serializable_globals = {{k: v for k, v in globals().items() if is_serializable(v)}}

with open("{result_file}", "w+") as result_file:
    json.dump(serializable_globals, result_file)
        """
            file.write(write_namespace_code)

        result = self._execute_file(program_executable_path)
        if result:
            error, stacktrace = result
        else:
            error = None
            stacktrace = None

        if error:
            error = str(error)
            stacktrace = str(stacktrace)

            method_name = re.compile(r"def\s+(\w+)\s*\(").search(api_info["signature"]).group(1)
            print(f"Error in executing {method_name}: {error}")

            undefined_method = re.search(r"name '(\w+)' is not defined", error)
            if undefined_method:
                undefined_method = undefined_method.group(1)
                try:
                    if (
                        len(self.method_stack) > 4
                        and self.method_stack[-2] == undefined_method
                        and self.method_stack[-3] == method_name
                        and self.method_stack[-4] == undefined_method
                    ):
                        print("Infinite recursion detected")
                        self.error_counts[
                            self.signature_agent.method_names.index(undefined_method)
                        ] = self.max_num_errors
                        self.error_counts[
                            self.signature_agent.method_names.index(method_name)
                        ] = self.max_num_errors
                        api_info["error"] = "Implementation failed"
                        return api_info
                    elif undefined_method == method_name:
                        self.error_counts[
                            self.signature_agent.method_names.index(method_name)
                        ] = self.max_num_errors
                        api_info["error"] = "Implementation failed"
                        return api_info

                    method_name_index = self.signature_agent.method_names.index(
                        undefined_method
                    )
                    header = self.signature_agent.headers[method_name_index]
                    self.method_stack.append(undefined_method)
                    self(
                        undefined_method,
                        header["docstring"],
                        header["signature"],
                        results_folder_path,
                        prompt=prompt,
                    )
                    return self.test_implementation(
                        api_info, results_folder_path, prompt=prompt
                    )
                except ValueError:
                    self.error_counts[
                        self.signature_agent.method_names.index(method_name)
                    ] += 1
                    api_info["error"] = "Implementation failed"
                    return api_info
            else:
                self.error_counts[
                    self.signature_agent.method_names.index(method_name)
                ] += 1
                api_info["error"] = stacktrace
                return api_info
        self.implemented[self.signature_agent.method_names.index(method_name)] = True
        api_info["error"] = None
        return api_info

    def _trace_execution(self, frame, event, arg):
        if event == "line":
            filename = frame.f_globals.get("__file__", None)
            if filename:
                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                if lineno > self.namespace_line:
                    return self._trace_execution
                if "import math" in line:
                    return self._trace_execution
                if "import" in line:
                    self.namespace_line = lineno
                    return self._trace_execution
                with open(self.trace_file_path, "a+") as f:
                    f.write(f"<p>{lineno}: {line}</p>\n")
        return self._trace_execution

    def _execute_file(self, program_executable_path):
        sys.settrace(self._trace_execution)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(60)
            runpy.run_path(program_executable_path, init_globals=self.namespace)
            signal.alarm(0)
        except TimeoutException as e:
            stacktrace = traceback.format_exc()
            return e, stacktrace
        except Exception as e:
            stacktrace = traceback.format_exc()
            return e, stacktrace
        finally:
            sys.settrace(None)
        return

    @staticmethod
    def _parse_signature_params(signature):
        match = re.search(r"def \w+\(([^)]*)\)", signature)
        if not match:
            return []
        params_str = match.group(1).strip()
        if not params_str:
            return []
        param_types = []
        for param in params_str.split(","):
            param = param.strip()
            if not param:
                continue
            parts = param.split(":")
            name = parts[0].strip()
            ptype = parts[1].strip() if len(parts) > 1 else "str"
            param_types.append((name, ptype))
        return param_types

    def _get_docstring_types(self, docstring, signature=None):
        arg_types = self._parse_signature_params(signature) if signature else []

        if not arg_types:
            args_pattern = re.compile(r"Args:\s*((?:\s+\w+ \(\w+\): .+\n)+)")
            args_match = args_pattern.search(docstring)
            args_section = args_match.group(1) if args_match else ""
            arg_types = re.findall(r"\s+(\w+) \((\w+)\):", args_section)

        returns_pattern = re.compile(r"Returns:\s+(\w+): .+")
        returns_match = returns_pattern.search(docstring)
        returns_section = returns_match.group(1) if returns_match else ""

        return arg_types, returns_section

    def _get_return_code(self, returns):
        if self.dataset in ["clevr", "gqa"]:
            if returns == "string":
                return '\n\treturn ""'
            elif returns == "image":
                return "\n\treturn image"
            elif returns == "int":
                return "\n\treturn 25"
            elif returns == "float":
                return "\n\treturn 1.0"
            elif returns == "list":
                return "\n\treturn [[25, 25]]"
            elif returns == "bool":
                return "\n\treturn False"
            elif returns == "tuple":
                return "\n\treturn (50, 50)"
            else:
                return "\n\treturn 1"
        else:
            if returns == "string":
                return '\n\treturn ""'
            elif returns == "image":
                return "\n\treturn image"
            elif returns == "int":
                return "\n\treturn 25"
            elif returns == "float":
                return "\n\treturn 1.0"
            elif returns == "list":
                return "\n\treturn [[25, 25, 50, 50]]"
            elif returns == "bool":
                return "\n\treturn False"
            elif returns == "tuple":
                return "\n\treturn (50, 50)"
            else:
                return "\n\treturn 1"

    def get_api_implementations(self, results_folder_path, prompt=None, critic_report=None):
        if prompt is None:
            prompt = API_PROMPT

        headers = self.signature_agent.headers
        self._sync_signature_state()

        folder_name = "api_generator"
        results_folder_path = os.path.join(
            results_folder_path,
            f"{folder_name}",
        )
        os.makedirs(results_folder_path)
        self._results_folder = results_folder_path

        file_path = os.path.join(results_folder_path, "api_implementation.html")

        for header in headers:
            
            implementation, output = self(
                header["method_name"],
                header["docstring"],
                header["signature"],
                results_folder_path,
                prompt=prompt,
                critic_report=critic_report,
            )

            if self.write_results:

                method_name = (
                    re.compile(r"def\s+(\w+)\s*\(").search(header["signature"]).group(1)
                )
                implementation_results_path = os.path.join(
                    results_folder_path, f"{method_name}"
                )
                if not os.path.exists(implementation_results_path):
                    continue
                with open(os.path.join(implementation_results_path, f"{method_name}.html"), "wb+") as file:

                    generated_signatures = [
                        h["docstring"] + h["signature"]
                        for h in self.signature_agent.headers
                        if h["method_name"] != method_name
                    ]
                    generated_signatures = "\n\n".join(generated_signatures)
                    rag_tools = ""
                    if self.local_rag:
                        rag_tools = self.local_rag.prompt_context(
                            include_pending=True,
                            exclude_primitives=True,
                        )

                    file.write((f"<h1>Signature</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{header['docstring'] + header['signature']}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>Prompt</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{prompt.format(primitives=self.signature_agent._primitive_context(), predef_signatures=self.signature_agent.predef_signatures, generated_signatures=generated_signatures, rag_tools=rag_tools, docstring=header['docstring'], signature=header['signature'], critic_report=json.dumps(critic_report, ensure_ascii=False) if critic_report else '')}</code>\n".replace(
                                "\n", "<br>"
                            )
                        ).encode("utf-8")
                    )

                    file.write((f"<h1>LLM Output</h1>\n").encode("utf-8"))
                    file.write(
                        (f"<code>{output}</code>\n".replace("\n", "<br>")).encode(
                            "utf-8"
                        )
                    )

                    file.write((f"<h1>Implementation</h1>\n").encode("utf-8"))
                    file.write(
                        (
                            f"<code>{implementation}</code>\n".replace("\n", "<br>")
                        ).encode("utf-8")
                    )
                    file.close()
            self.pbar.update(1)
        self.pbar.close()

        api_path = os.path.join(results_folder_path, "api.json")

        with open(api_path, "w", encoding="utf-8") as file:
            json.dump(self.api, file, indent=2, ensure_ascii=False)

        return api_path, self.api


class CriticAgent(Agent):

    def __init__(self, model_name="deepseek-reasoner", api_key_path="./api.key"):
        super().__init__(model_name, write_results=False, api_key_path=api_key_path)

    def __call__(self, primitive_context, docstring, signature, implementation, review_history=None):
        history_block = ""
        if review_history:
            prev_entries = "\n".join(
                f"  Attempt {i}: issues={e['issues']}, suggestion={e['suggestion']}"
                for i, e in enumerate(review_history, 1)
            )
            history_block = f"""
===== PREVIOUS REVIEWS =====
{prev_entries}
If suggestions above alternate between two opposite answers, the implementation
is likely correct and the docstring is ambiguous. Mark is_correct: true.
"""

        review_prompt = f"""Verify this implementation against the primitives.

===== PRIMITIVES (authoritative) =====
{primitive_context}
{history_block}
===== FUNCTION =====
Docstring: {docstring}
Signature: {signature}
Implementation:
```python
{signature}
{implementation}
```

RULES:
1. If the implementation is correct, set is_correct: true with empty issues
   and suggestion. Only flag errors you are certain about.
2. The PRIMITIVES above define the ground truth for field names, indices, and
   types. Verify field accesses into primitive return values are correct.
3. Check that the computation logic matches the docstring's semantic intent.
4. Check return type matches the signature.

Output ONLY a JSON object:
{{"is_correct": true/false, "issues": "brief errors, or empty", "suggestion": "how to fix the IMPLEMENTATION, or empty"}}
"""
        output, _ = self.generator.generate(review_prompt)
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            try:
                result = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                return {"is_correct": True, "issues": "", "suggestion": ""}
        if not isinstance(result, dict):
            return {"is_correct": True, "issues": "", "suggestion": ""}
        result.setdefault("is_correct", True)
        return result

    def review_signature(self, primitive_context, docstring, signature, purpose="", review_history=None):
        """Review a generated signature/purpose/docstring for consistency with primitives.

        This method OWNS the docstring and purpose — it should correct them if they
        conflict with primitives. Signature changes are a last resort.
        """
        if not primitive_context:
            return {"is_correct": True, "issues": "", "suggestion": ""}

        history_block = ""
        if review_history:
            prev = "\n".join(
                f"  Attempt {i}: issues={e['issues']}, suggestion={e['suggestion']}"
                for i, e in enumerate(review_history, 1)
            )
            history_block = f"""
===== PREVIOUS REVIEWS =====
{prev}
If suggestions alternate between two opposite answers, the signature is likely
correct and the docstring/purpose need fixing. Flag the docstring, not the signature.
"""

        review_prompt = f"""Verify this function spec against the primitives.

===== PRIMITIVES (authoritative) =====
{primitive_context}
{history_block}
===== FUNCTION TO REVIEW =====
Purpose: {purpose}
Docstring: {docstring}
Signature: {signature}

RULES:
1. Check that field names, indices, and types referenced in the docstring are
   consistent with the primitives above.
2. Check docstring Args match signature parameters (count, names, types).
3. Check the Returns type matches the signature return type.
4. Prefer fixing the PURPOSE or DOCSTRING. Only suggest a signature change
   if the parameters or return type are truly wrong.

Output ONLY a JSON object:
{{"is_correct": true/false, "issues": "brief description of any issues found, or empty if correct", "suggestion": "how to fix, or empty if correct"}}
"""
        output, _ = self.generator.generate(review_prompt)
        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", output, re.DOTALL)
            try:
                result = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                return {"is_correct": True, "issues": "", "suggestion": ""}
        if not isinstance(result, dict):
            return {"is_correct": True, "issues": "", "suggestion": ""}
        result.setdefault("is_correct", True)
        return result


class VLMAgent(Agent):
    def __init__(
        self,
        model_name="deepseek-reasoner",
        write_results=True,
        dataset="omni3d",
        local_rag=None,
        include_pending_tools=False,
        coordinate_scale=1000,
        log_dir="vlm_agent_logs",
        generator_base_url=None,
        generator_api_key=None,
        generator_extra_body=None,
        generator_temperature=None,
    ):
        super().__init__(model_name, write_results, dataset=dataset)
        self.local_rag = local_rag
        self.include_pending_tools = include_pending_tools
        self.coordinate_scale = coordinate_scale  
        self.log_dir = log_dir

        if generator_base_url is not None or generator_api_key is not None:
            self.generator = Generator(
                model_name=model_name,
                base_url=generator_base_url or "https://api.deepseek.com",
                api_key=generator_api_key,
                api_key_path=None,
                extra_body=generator_extra_body,
            )
        if generator_temperature is not None:
            self.generator.temperature = generator_temperature

        self._tool_registry: Dict[str, callable] = {}
        self._tool_schemas: List[Dict[str, Any]] = []
        self._tools_loaded = False


    def _load_tools(self):

        if self._tools_loaded:
            return

        import importlib
        import inspect

        namespace: Dict[str, Any] = {}

        try:
            from agents.functions.primitive import PRIMITIVES_REGISTRY

            namespace.update(PRIMITIVES_REGISTRY)
        except Exception:
            pass

        try:
            import agents.functions.addition as addition_module

            for fname, func in inspect.getmembers(addition_module, inspect.isfunction):
                if fname.startswith("_"):
                    namespace[fname] = func
        except Exception:
            pass

        rag_entries = []
        if self.local_rag:
            rag_entries = self.local_rag.retrieve(
                include_pending=self.include_pending_tools,
                exclude_primitives=True,
            )

        for entry in rag_entries:
            name = entry.name
            if name in namespace:
                self._tool_registry[name] = namespace[name]
            elif name in self._tool_registry:
                pass

        if not self._tool_registry:
            for fname, func in namespace.items():
                if fname.startswith("_"):
                    self._tool_registry[fname] = func

        self._tool_schemas = self._build_tool_schemas(rag_entries)
        self._tools_loaded = True

    def _build_tool_schemas(
        self, rag_entries: List[Any]
    ) -> List[Dict[str, Any]]:
        import inspect

        seen_names = set()
        schemas = []

        for entry in rag_entries:
            name = entry.name
            if name not in self._tool_registry:
                continue
            seen_names.add(name)
            schema = self._entry_to_schema(entry)
            if schema:
                schemas.append(schema)

        for tname in self._tool_registry:
            if tname not in seen_names:
                schema = self._function_to_schema(tname, self._tool_registry[tname])
                if schema:
                    schemas.append(schema)

        return schemas

    def _entry_to_schema(self, entry: Any) -> Optional[Dict[str, Any]]:
        name = entry.name
        description = (
            getattr(entry, "purpose", None)
            or getattr(entry, "docstring", "")
        )
        if description and hasattr(description, "strip"):
            description = description.strip()
        # Extract just the first meaningful sentence
        if description:
            lines = description.splitlines()
            for line in lines:
                stripped = line.strip()
                if stripped and not any(
                    stripped.startswith(p)
                    for p in ("Args:", "Returns:", "Raises:", "Note:", "Example:", "Internally")
                ):
                    description = stripped
                    break

        params = self._infer_parameters(name)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description or f"Call the {name} function.",
                "parameters": params,
            },
        }

    def _function_to_schema(
        self, name: str, func: callable
    ) -> Optional[Dict[str, Any]]:
        """Build a schema from a bare Python function without a RAG entry."""
        import inspect

        doc = inspect.getdoc(func) or ""
        description = doc.splitlines()[0] if doc else f"Call the {name} function."

        params = self._infer_parameters(name, func)
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": params,
            },
        }

    def _infer_parameters(
        self, name: str, func: callable = None
    ) -> Dict[str, Any]:
        import inspect

        properties = {}
        required = []

        if func is None:
            func = self._tool_registry.get(name)
        if func is None:
            return {"type": "object", "properties": properties, "required": required}

        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError):
            return {"type": "object", "properties": properties, "required": required}

        for pname, param in sig.parameters.items():
            if pname == "image_path":
                continue
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                json_type = "string"
            elif ann in (int, "int"):
                json_type = "integer"
            elif ann in (float, "float"):
                json_type = "number"
            elif ann in (list, "list"):
                json_type = "array"
            elif ann in (bool, "bool"):
                json_type = "boolean"
            else:
                json_type = "string"

            properties[pname] = {"type": json_type}
            required.append(pname)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _denormalize_coords(self, coords: list, img_w: int, img_h: int) -> list:
        """Convert a coordinate list from normalized scale to pixel coordinates.

        Even indices are x-coordinates (scaled by width), odd indices are
        y-coordinates (scaled by height). Handles bbox [x1,y1,x2,y2],
        point [x,y], and any other flat coordinate list.
        """
        s = self.coordinate_scale
        if not s:
            return coords
        result = []
        for i, v in enumerate(coords):
            if i % 2 == 0:  
                result.append(int(v * img_w / s))
            else:          
                result.append(int(v * img_h / s))
        return result

    def _normalize_coords(self, coords: list, img_w: int, img_h: int) -> list:
        s = self.coordinate_scale
        if not s:
            return coords
        result = []
        for i, v in enumerate(coords):
            if i % 2 == 0:  
                result.append(round(v * s / img_w))
            else:     
                result.append(round(v * s / img_h))
        return result


    def _execute_tool_call(
        self, call: Dict[str, Any], image_path: str, img_w: int, img_h: int
    ) -> Dict[str, Any]:
        tool_name = call.get("name", "")
        parameters = dict(call.get("parameters", {}))

        if tool_name not in self._tool_registry:
            return {
                "success": False,
                "error": f"Tool not found: {tool_name}",
            }

        for key in list(parameters.keys()):
            val = parameters[key]
            if not isinstance(val, list) or not val:
                continue
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in val):
                parameters[key] = self._denormalize_coords(val, img_w, img_h)

        parameters["image_path"] = image_path

        try:
            func = self._tool_registry[tool_name]
            result = func(**parameters)
            return {
                "success": True,
                "result": result,
                "description": self._format_tool_result(tool_name, parameters, result),
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    def _format_tool_result(
        self, name: str, parameters: Dict[str, Any], result: Any
    ) -> str:
        """Produce a human-readable description of a tool execution result."""
        params_summary = {
            k: v for k, v in parameters.items() if k != "image_path"
        }
        desc = f"Tool '{name}' called with {json.dumps(params_summary, default=str)}.\nResult: "
        if isinstance(result, (int, float)):
            desc += f"{result:.3f} meters" if isinstance(result, float) else f"{result} meters"
        elif isinstance(result, list) and result and isinstance(result[0], list):
            flat = result[0] if len(result) == 1 else result
            desc += json.dumps(flat, default=str)
        else:
            desc += json.dumps(result, default=str)
        return desc


    def _generate_multimodal_response(self, image_path: str, prompt: str) -> str:
        """Send a single multimodal (image + text) request to the VLM."""
        return self.generator.single_image_inference(image_path, prompt)

    @staticmethod
    def _parse_tool_calls(response: str) -> List[Dict[str, Any]]:
        tool_calls = []
        pattern = r"<tool_call>\s*({.*?})\s*</tool_call>"
        for match in re.findall(pattern, response, re.DOTALL):
            try:
                tool_call = json.loads(match)
            except json.JSONDecodeError:
                continue
            if "name" in tool_call and "parameters" in tool_call:
                tool_calls.append(tool_call)
        return tool_calls

    @staticmethod
    def _extract_answer(response: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()


    def solve_problem(
        self,
        image_path: str,
        question: str,
        max_iterations: int = 6,
        **generator_kwargs,
    ) -> Dict[str, Any]:

        from pathlib import Path
        import time as _time

        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        self._load_tools()

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        iteration_responses: List[Dict[str, Any]] = []

        from prompts.vlm_tool_prompt import (
            create_system_prompt,
            create_user_prompt,
            create_follow_up_prompt,
            create_fallback_prompt,
        )

        system_prompt = create_system_prompt(
            self._tool_schemas, self.coordinate_scale
        )
        
        tool_use = bool(self._tool_schemas)
        user_prompt = create_user_prompt(question, [image_path], tool_use, self.coordinate_scale)

        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: List[Dict[str, Any]] = []
        all_responses: List[str] = []
        cumulative_history = ""
        current_tool_results: Dict[str, Any] = {}
        last_response: Optional[str] = None
        final_answer: Optional[str] = None
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1

            if iteration == 1:
                prompt = system_prompt + "\n\n" + user_prompt
            else:
                follow_up = create_follow_up_prompt(
                    question=question,
                    cumulative_history=cumulative_history,
                    tool_results=current_tool_results,
                    original_images=[image_path],
                    remaining_iterations=max_iterations - iteration + 1,
                )
                prompt = system_prompt + "\n\n" + follow_up
            
            response = self._generate_multimodal_response(image_path, prompt)
            all_responses.append(response)
            last_response = response

            iteration_responses.append(
                {"iteration": iteration, "response": response}
            )

            tool_calls = self._parse_tool_calls(response)
            has_answer = "<answer>" in response and "</answer>" in response

            if len(tool_calls) > 1:
                tool_calls = tool_calls[:1]

            if not tool_calls:
                if has_answer or iteration == max_iterations:
                    break
                current_tool_results = {
                    "System Notice": {
                        "success": False,
                        "error": (
                            "No tools called. You MUST use <tool_call> to "
                            "gather information or <answer> to finish."
                        ),
                    }
                }
                continue

            current_tool_results = {}
            for call in tool_calls:
                tname = call.get("name", "unknown")
                result = self._execute_tool_call(call, image_path, img_w, img_h)
                current_tool_results[tname] = result
                all_tool_results.append({"tool_call": call, "result": result})

            all_tool_calls.extend(tool_calls)

            history_entry_parts = [f"\n--- Step {iteration} ---"]
            history_entry_parts.append(f"Response: {response}")
            for tname, r in current_tool_results.items():
                if r.get("success"):
                    history_entry_parts.append(
                        f"Tool {tname}: {r.get('description', str(r))}"
                    )
                else:
                    history_entry_parts.append(
                        f"Tool {tname} ERROR: {r.get('error', 'unknown')}"
                    )
            cumulative_history += "\n".join(history_entry_parts) + "\n"

        if last_response and not self._extract_answer(last_response):
            fallback_prompt_text = create_fallback_prompt(
                question, cumulative_history
            )
            fallback_response = self._generate_multimodal_response(
                image_path,
                system_prompt + "\n\n" + fallback_prompt_text,
            )
            all_responses.append(fallback_response)
            last_response = fallback_response

        final_answer = (
            self._extract_answer(last_response or "")
            or last_response
            or ""
        )

        return {
            "answer": final_answer,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iteration,
            "responses": all_responses,
            "iteration_logs": iteration_responses,
        }

    def run(
        self,
        questions: List[Dict[str, Any]],
        images_folder_path: str,
        results_folder_path: str,
        max_iterations: int = 6,
    ) -> List[Dict[str, Any]]:
        import time as _time

        os.makedirs(results_folder_path, exist_ok=True)
        results = []
        all_iteration_logs: List[Dict[str, Any]] = []

        for qdata in tqdm(questions, desc="VLMAgent"):
            image_filename = qdata.get("image_filename", "")
            image_path = os.path.join(images_folder_path, image_filename)
            question_text = qdata.get("question", "")

            try:
                result = self.solve_problem(
                    image_path=image_path,
                    question=question_text,
                    max_iterations=max_iterations,
                )
                
                result["question"] = qdata
                result["image_path"] = image_path
                all_iteration_logs.append({
                    "image_path": image_path,
                    "question": question_text,
                    "max_iterations": max_iterations,
                    "answer": result.get("answer", ""),
                    "tool_calls": result.get("tool_calls", []),
                    "tool_results": result.get("tool_results", []),
                    "iterations": result.get("iterations", 0),
                    "responses": result.pop("iteration_logs", []),
                })
                
            except Exception as exc:
                result = {
                    "question": qdata,
                    "image_path": image_path,
                    "answer": "",
                    "error": str(exc),
                    "tool_calls": [],
                    "tool_results": [],
                    "iterations": 0,
                    "responses": [],
                }
                all_iteration_logs.append({
                    "image_path": image_path,
                    "question": question_text,
                    "max_iterations": max_iterations,
                    "answer": "",
                    "tool_calls": [],
                    "tool_results": [],
                    "iterations": 0,
                    "error": str(exc),
                    "responses": [],
                })
            results.append(result)

        out_path = os.path.join(results_folder_path, "vlm_agent_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = _time.strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(
            self.log_dir, f"{self.dataset}_{timestamp}.json"
        )
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(all_iteration_logs, f, ensure_ascii=False, indent=2, default=str)

        print(f"Results saved to: {out_path}")
        print(f"Iteration logs saved to: {log_filename}")
        return results
