import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
import json
from agents.agents import (
    FuncAgent,
    SignatureAgent,
    VLMAgent,
)
from agents.local_rag import LocalRAG
from prompts.modules import MODULES_SIGNATURES


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_questions(questions_data, num_questions):
    questions = list(questions_data["questions"])
    if num_questions is None or num_questions < 0:
        return questions
    return questions[:num_questions]


def convert_rgpt_data(rgpt_path):
    import re

    with open(rgpt_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    questions = []
    for idx, item in enumerate(raw_data):
        text_q = item.get("text_q", "").strip()

        answer = ""
        for conv in item.get("conversations", []):
            if conv.get("from") == "gpt":
                answer = conv.get("value", "").strip()
                break

        file_path = item.get("file_path", "")
        image_filename = os.path.basename(file_path)

        questions.append(
            {
                "question": text_q,
                "answer": answer,
                "image_filename": image_filename,
                "image_index": idx,
                "question_index": idx,
            }
        )

    return {"questions": questions}


def convert_cvbench_data(cvbench_path):
    questions = []
    with open(cvbench_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)

            text_q = item.get("question", "").strip()

            answer = item.get("answer", "").strip()

            image_filename = item.get("filename", "")

            questions.append(
                {
                    "question": text_q,
                    "answer": answer,
                    "image_filename": image_filename,
                    "image_index": idx,
                    "question_index": idx,
                }
            )

    return {"questions": questions}


def convert_ibims_data(ibims_path):
    with open(ibims_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    questions = []
    for idx, item in enumerate(raw_data):
        text_q = item.get("prompt", "").strip()
        gt_depth = item.get("gt_depth")
        if gt_depth is not None:
            answer = f"{gt_depth} meters"
        else:
            answer = ""
        image_filename = item.get("image", "")

        questions.append(
            {
                "question": text_q,
                "answer": answer,
                "image_filename": image_filename,
                "image_index": idx,
                "question_index": idx,
            }
        )

    return {"questions": questions}


def convert_blink_data(blink_path):
    with open(blink_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    questions = []
    for idx, item in enumerate(raw_data):
        text_q = item.get("prompt", "").strip()
        answer = item.get("ground_truth", "").strip()
        image_filename = item.get("image", "")

        questions.append(
            {
                "question": text_q,
                "answer": answer,
                "image_filename": image_filename,
                "image_index": idx,
                "question_index": idx,
            }
        )

    return {"questions": questions}


def convert_qspatial_data(qspatial_path):
    questions = []
    with open(qspatial_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)

            text_q = item.get("question", "").strip()
            answer_value = item.get("answer_value")
            answer_unit = item.get("answer_unit", "")

            if answer_value is not None:
                answer = f"{answer_value} {answer_unit}".strip()
            else:
                answer = ""

            image_filename = item.get("image_rel_path", "")

            questions.append(
                {
                    "question": text_q,
                    "answer": answer,
                    "image_filename": image_filename,
                    "image_index": idx,
                    "question_index": idx,
                }
            )

    return {"questions": questions}


def load_annotations(args):
    if args.dataset == "rgpt":
        return convert_rgpt_data(args.annotations_json)
    if args.dataset == "cvbench":
        return convert_cvbench_data(args.annotations_json)
    if args.dataset == "ibims":
        return convert_ibims_data(args.annotations_json)
    if args.dataset == "blink":
        return convert_blink_data(args.annotations_json)
    if args.dataset == "qspatial":
        return convert_qspatial_data(args.annotations_json)
    with open(args.annotations_json, "r") as file:
        return json.load(file)


def get_predefined_signatures():
    return MODULES_SIGNATURES


def read_text_if_exists(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def run_rag_build_agents(args):
    """Generate additional functions and persist them into LocalRAG only."""
    questions_data = load_annotations(args)
    questions = select_questions(questions_data, args.num_questions)
    if not questions:
        raise ValueError("No questions loaded for RAG build workflow")

    local_rag = LocalRAG(registry_path=args.rag_registry_path)
    signature_agent = SignatureAgent(
        get_predefined_signatures(args.dataset),
        dataset=args.dataset,
        local_rag=local_rag,
    )
    func_agent = FuncAgent(
        signature_agent,
        args.dataset,
        local_rag=local_rag,
    )

    results_folder_path = os.path.join(
        args.results_pth, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(results_folder_path, exist_ok=True)
    
    api_questions = list(questions) if args.num_api_questions < 0 else questions[:args.num_api_questions]

    print("[RAG Build] Generating function signatures...")
    signature_agent.get_signatures(
        api_questions,
        args.image_pth,
        results_folder_path,
        include_pending_tools=args.include_pending_tools,
    )

    print("[RAG Build] Generating and testing implementations...")
    func_path, func_info = func_agent.get_func_implementations(results_folder_path)

    rag_snapshot_path = os.path.join(results_folder_path, "local_rag_snapshot.json")
    with open(rag_snapshot_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "registry_path": str(local_rag.registry_path),
                "addition_path": str(local_rag.addition_path),
                "generated_func_path": func_path,
                "generated_functions": [item["method_name"] for item in func_info],
                "rag_entries": [
                    {
                        "name": entry.name,
                        "status": entry.status,
                        "source": entry.source,
                        "signature": entry.signature,
                    }
                    for entry in local_rag.entries.values()
                ],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[RAG Build] Generated functions saved to: {local_rag.addition_path}")
    print(f"[RAG Build] RAG registry saved to: {local_rag.registry_path}")
    print(f"[RAG Build] Run snapshot saved to: {rag_snapshot_path}")  
    

def run_vlm_agent(args):
    """Run the VLMAgent — standalone multi-turn tool-calling with generated functions."""
    questions_data = load_annotations(args)
    questions = select_questions(questions_data, args.num_questions)
    if not questions:
        raise ValueError("No questions loaded for VLMAgent workflow")

    local_rag = LocalRAG(registry_path=args.rag_registry_path)

    if args.vlm_model_backend == "local":
        agent = VLMAgent(
            model_name=args.local_model_name,
            dataset=args.dataset,
            local_rag=local_rag,
            include_pending_tools=args.include_pending_tools,
            coordinate_scale=args.coordinate_scale,
            generator_base_url=args.local_model_url,
            generator_api_key="EMPTY",
            generator_extra_body=None,
            generator_temperature=args.vlm_temperature,
        )
    elif args.vlm_model_backend == "qwen":
        agent = VLMAgent(
            model_name=args.qwen_model_name,
            dataset=args.dataset,
            local_rag=local_rag,
            include_pending_tools=args.include_pending_tools,
            coordinate_scale=args.coordinate_scale,
            generator_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            generator_api_key=args.qwen_api_key,
            generator_extra_body={"enable_thinking": False},
            generator_temperature=args.vlm_temperature,
        )
    else:
        raise ValueError(f"Unsupported VLM model backend: {args.vlm_model_backend}")

    results_folder_path = os.path.join(
        args.results_pth, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(results_folder_path, exist_ok=True)

    print(f"[VLMAgent] Running {len(questions)} questions...")
    agent.run(
        questions=questions,
        images_folder_path=args.image_pth,
        results_folder_path=results_folder_path,
        max_iterations=args.execution_max_iterations,
    )


if __name__ == "__main__":
    set_seeds(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workflow",
        default="vlm_agent",
        choices=["rag_build", "vlm_agent"],
        help=(
            "Workflow to run: RAG build only, original VADAR, full evolution, "
            "VLM agent standalone, or VLM + automated critic loop "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dataset",
        default="qspatial",
        choices=["rgpt", "cvbench", "ibims", "blink", "qspatial"],
        help="Name of dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--annotations-json",
        default="",
        help="Path to JSON file of questions (default: %(default)s)",
    )
    parser.add_argument(
        "--image-pth",
        default="",
        help="Path to directory containing images associated with questions (default: %(default)s)",
    )
    parser.add_argument(
        "--models-path",
        default="models/",
        help="Path to directory containing models (default: %(default)s)",
    )
    parser.add_argument(
        "--results-pth",
        default="results/",
        help="Path to directory to save html results in (default: %(default)s)",
    )
    parser.add_argument(
        "--scenes-json",
        default="",
        help="Path to json file with scene data per image (default: %(default)s)",
    )
    parser.add_argument(
        "--num-questions",
        default=-1,
        type=int,
        help="Number of questions to loop through. -1 for all. Must be at least num_api_questions (default: %(default)s)",
    )
    parser.add_argument(
        "--num-api-questions",
        default=-1,
        type=int,
        help="Number of questions to use for API generation. -1 for all. (default: %(default)s)",
    )
    parser.add_argument(
        "--rag-registry-path",
        default="",
        help="Path to LocalRAG JSON registry (default: agents/functions/local_rag_registry.json)",
    )
    parser.add_argument(
        "--outer-loop-iters",
        default=5,
        type=int,
        help="Number of evolution execution/critic refinement loops (default: %(default)s)",
    )
    parser.add_argument(
        "--include-pending-tools",
        default=True,
        help="Allow pending generated tools in retrieval during outer-loop integration",
    )
    parser.add_argument(
        "--execution-max-iterations",
        default=6,
        type=int,
        help="Maximum multimodal tool-calling turns per question (default: %(default)s)",
    )
    parser.add_argument(
        "--vlm-model-backend",
        default="qwen",
        choices=["local", "qwen"],
        help="VLM backend: local vLLM, or Qwen DashScope API (default: %(default)s)",
    )
    parser.add_argument(
        "--local-model-name",
        default="qwen3-vl-4b-instruct",
        help="Model name for local vLLM backend (default: %(default)s)",
    )
    parser.add_argument(
        "--local-model-url",
        default="http://localhost:8000/v1",
        help="Base URL for local vLLM backend (default: %(default)s)",
    )
    parser.add_argument(
        "--qwen-api-key",
        default="",
        help="DashScope API key for Qwen backend",
    )
    parser.add_argument(
        "--qwen-model-name",
        default="qwen3.6-flash", # qwen3.6-flash   qwen3.6-plus
        help="Model name for Qwen DashScope backend (default: %(default)s)",
    )
    parser.add_argument(
        "--vlm-temperature",
        default=0.7,
        type=float,
        help="VLM sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--coordinate-scale",
        default=0,
        type=int,
        help="Coordinate scale for bbox normalization (0 = disabled, raw pixel coords). Set to 1000 for normalized coords. (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.workflow == "rag_build":
        run_rag_build_agents(args)
    else:
        run_vlm_agent(args)
