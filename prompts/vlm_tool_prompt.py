"""
Prompt templates for VLMAgent — the VLM-led multi-turn tool-calling agent.

Mirrors the SPAgentLLMStyle prompt pattern (system → user → follow_up → fallback)
but adapted for VADAR's generated spatial-reasoning function tools.
"""

from typing import List, Dict, Any, Optional
import json


def _build_tool_description_blocks(tool_schemas: List[Dict[str, Any]]) -> str:
    """Build human-readable tool description blocks from JSON schemas."""
    if not tool_schemas:
        return "No tools available."

    blocks = []
    for i, schema in enumerate(tool_schemas):
        func = schema.get("function", schema)
        name = func.get("name", "unknown")
        desc = func.get("description", "No description.")
        params = func.get("parameters", {}).get("properties", {})

        param_lines = []
        for param_name, param_info in params.items():
            if param_name == "image_path":
                continue
            ptype = param_info.get("type", "string")
            pdesc = param_info.get("description", "")
            param_lines.append(f"    - `{param_name}` ({ptype}): {pdesc}")

        param_str = "\n".join(param_lines) if param_lines else "    (no additional parameters)"

        # Build an illustrative example
        example_params = {}
        for param_name, param_info in params.items():
            if param_name == "image_path":
                continue
            ptype = param_info.get("type", "string")
            if ptype == "array":
                example_params[param_name] = [250, 250, 750, 750]
            elif ptype in ("integer", "number"):
                example_params[param_name] = 500
            else:
                example_params[param_name] = "value"

        example_json = json.dumps({"name": name, "parameters": example_params})

        blocks.append(
            f"## {i + 1}. {name}\n"
            f"* **Function:** {desc}\n"
            f"* **Parameters:**\n{param_str}\n"
            f"* **Example:**\n"
            f"<tool_call>\n{example_json}\n</tool_call>"
        )

    return "\n\n".join(blocks)


def create_system_prompt(tool_schemas: List[Dict[str, Any]], coordinate_scale: int = 1000) -> str:
    tool_blocks = _build_tool_description_blocks(tool_schemas)

    if coordinate_scale:
        coordinate_section = f"""------
# Coordinate System
All bounding box coordinates use a normalized 0-{coordinate_scale} coordinate system.
- (0, 0) is the top-left corner of the image.
- ({coordinate_scale}, {coordinate_scale}) is the bottom-right corner.
- A bounding box is [x_min, y_min, x_max, y_max] in this normalized space.
"""
    else:
        coordinate_section = ""

    return f"""You are an elite visual-spatial reasoning agent.
You WILL be given one or more images. You should use tools to obtain precise spatial measurements or additional evidence when needed.

# Available Tools
You have access to the following tools to assist with user questions:

{tool_blocks}

{coordinate_section}------
# How to call a tool
When you need to use a tool, return a strictly valid JSON object with the function name and parameters within <tool_call></tool_call> tags:

<tool_call>
{{"name": "<function-name>", "parameters": {{"key1": value1, "key2": value2}}}}
</tool_call>

Replace the placeholders (<function-name>, key, value) with the actual tool name and the required parameter keys and values.

**NOTE:** The `image_path` parameter is passed automatically. Do NOT include `image_path` in your <tool_call> parameters.

# Critical constraint (must follow)
- NEVER output more than one <tool_call> block in a single response.
- Always place your step-by-step reasoning inside <think></think> tags before any tool call.
- When you have enough information to answer the question, output your final answer inside <answer></answer> tags.
- NEVER output <answer> and <tool_call> in the same response.
"""


def create_user_prompt(question: str, image_paths: List[str], tool_use: bool, coordinate_scale: int = 1000) -> str:
    images_info = "\n".join(f"- {path}" for path in image_paths)

    base_prompt = f"""=== TASK CONTEXT ===
User Question:
{question}

Available Images (use these exact paths if a tool needs image_path):
{images_info}
"""

    if tool_use:
        coord_note = (
            f"- Remember: all bounding boxes use the normalized 0-{coordinate_scale} coordinate system.\n"
            if coordinate_scale
            else ""
        )
        base_prompt += f"""
OUTPUT RULES (must follow):
- First, analyze the image visually and reason about what tools you need inside <think></think> tags.
- If you need a tool: output exactly ONE <tool_call>...</tool_call> block.
- If you have enough information to answer the question: output <answer>...</answer> only (no tool calls).
- NEVER output more than one <tool_call> block in a single response.
{coord_note}"""
    else:
        base_prompt += """
OUTPUT RULES (must follow):
- Analyze the image and your knowledge, then output <answer>...</answer> with your final answer.
"""

    return base_prompt


def create_follow_up_prompt(
    question: str,
    cumulative_history: str,
    tool_results: Dict[str, Any],
    original_images: List[str],
    remaining_iterations: int = 0,
) -> str:
    tool_summary = []
    for tool_name, result in tool_results.items():
        if result.get("success"):
            desc = result.get("description", json.dumps(result, ensure_ascii=False, default=str))
            tool_summary.append(f"- {tool_name} Returned:\n{desc}")
        else:
            tool_summary.append(f"- {tool_name} Failed: {result.get('error', 'Unknown error')}")

    images_info = "\n".join(f"- {path}" for path in original_images)

    return f"""=== TASK ===
Available Images (use these exact paths if a tool needs image_path):
{images_info}

Original Question: {question}

=== Existing tool call information ===
{cumulative_history}

--- The Latest Step Observations ---
{chr(10).join(tool_summary) if tool_summary else '(no tool results yet)'}

=== NEXT ACTION ===
You have {remaining_iterations} more iteration(s) available.

=== CRITICAL EXECUTION LOGIC ===
1. ANALYZE: Review the Original Question and the gathered information above.
2. THINK: Always place your step-by-step reasoning inside <think></think> tags.
3. NEXT ACTION: Based on your reasoning, choose EXACTLY ONE of the following actions:
   - [TOOL CALL]: IF more information is needed, output exactly ONE <tool_call></tool_call> block with the required parameters.
   - [Final Answer]: IF the existing information is sufficient to answer the Original Question, output your concise final answer inside <answer></answer> tags. (This terminates the task).
   *WARNING: Never output <answer> and <tool_call> in the same iteration.*
"""


def create_fallback_prompt(question: str, cumulative_history: str) -> str:
    return f"""=== SYSTEM OVERRIDE: FORCED FALLBACK ===
Original Question: {question}

=== INTERACTION HISTORY & OBSERVATIONS ===
{cumulative_history}

=== CRITICAL INSTRUCTION ===
You have reached the maximum iteration limit. TOOL USAGE IS NOW STRICTLY DISABLED.
You MUST NOT attempt to call any further tools.
Synthesize all the gathered information above and provide your best possible answer.
Place your concise final answer inside <answer></answer> tags.
"""
