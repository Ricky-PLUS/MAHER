CRITIC_PROMPT = """
You are a critic and router for a code-evolving visual reasoning system.

Given a failed validation case, inspect the question, wrong answer, ground truth,
tool-calling transcript, and execution trace. Attribute the failure to the earliest
tool layer that should be repaired.

Route using exactly one error_type:
1) "implementation_error": the function interface is usable, but its body has
   wrong logic, missing physical conversion, poor empty-result handling, or weak
   exception compatibility.
2) "signature_error": the function was designed with the wrong inputs, missing
   necessary context, wrong return type, or an interface that cannot support the
   question.
3) "unknown": the evidence is insufficient or the issue is outside generated
   functions.

Return only valid JSON with this schema:
{{
  "error_type": "implementation_error | signature_error | unknown",
  "reason": "brief concrete diagnosis",
  "suggestion": "actionable repair instruction for the routed agent",
  "target_function": "name of the most relevant generated function, or empty string"
}}

Question:
{question}

Image path:
{image_path}

Wrong answer:
{answer}

Ground truth:
{ground_truth}

Tool-calling transcript:
{program}

Execution trace:
{trace}
"""


CRITIC_VLM_PROMPT = """
You are a signature-level critic for a visual reasoning system that answers spatial
questions about images using dynamically generated Python functions.

Given a question the system answered incorrectly, the full tool-calling transcript,
and ALL available function signatures with docstrings, determine whether the error
originates from a poorly designed function **signature or docstring**.

Route using exactly one error_type:
1) "signature_error": the function's signature or docstring is wrong — missing
   parameters, wrong parameter types, wrong return type, ambiguous docstring that
   would mislead an implementer, or the interface fundamentally cannot support
   the question.
2) "implementation_error": the function's interface (signature + docstring) is
   correct and sufficient for the question, but the body logic must be flawed.
3) "unknown": cannot determine from the available evidence, or the issue lies
   outside the generated functions.

Analysis steps:
1. Read the question and identify what information is needed to answer it.
2. Examine which function(s) were called and what they returned.
3. Check whether each called function's signature provides the right inputs and
   return type for the question.
4. Check whether the docstring clearly and accurately describes what the function
   does, matching the signature's parameter names and types.
5. If a signature/docstring flaw explains the wrong answer, classify as
   signature_error and explain what to change.

Return only valid JSON with this schema:
{{
  "error_type": "signature_error | implementation_error | unknown",
  "reason": "brief concrete diagnosis referencing specific evidence from the transcript",
  "suggestion": "actionable repair instruction for SignatureAgent — describe what to change in the signature or docstring and why",
  "target_function": "name of the function whose signature/docstring needs repair, or empty string"
}}

Question:
{question}

Ground truth answer:
{ground_truth}

System's wrong answer:
{answer}

Full tool-calling transcript (tool calls, results, and LLM reasoning per iteration):
{vlm_transcript}

Available functions (ALL generated API functions with signatures and docstrings):
{function_registry}
"""


CRITIC_VLM_BATCH_PROMPT = """
You are a signature-level critic for a visual reasoning system that answers spatial
questions about images using dynamically generated Python functions.

You will receive TWO sections:
1. ALL SAMPLED QUESTIONS — every question the system needs to answer. This shows
   the FULL scope of question types (height, width, distance, etc.) the function
   library must support.
2. WRONG CASES — the subset of questions the system answered incorrectly, each
   with ground truth, wrong answer, and full tool-calling transcript.

Also provided are the MODIFIABLE generated functions (primitives are excluded —
they are authoritative low-level building blocks that cannot be changed).

Your job: analyze the wrong cases AS A WHOLE against the full question set. Ask:
- Is a function MISSING for one or more question types?
- Does an existing function have a functionally wrong signature or docstring?

CRITICAL: Before suggesting deletion or renaming of ANY function, check ALL
SAMPLED QUESTIONS to verify the function is not needed for other question types.
A function that appears to fail in wrong cases may still be essential for question
types that are NOT in the wrong cases. If a function is useful for some questions,
do NOT suggest removing it. Suggest ADDING a new function instead.

Output a SINGLE JSON object (NOT an array):

If the failures stem from function signature/docstring defects:
{{
  "error_type": "signature_error",
  "reason": "overall diagnosis — what is wrong with the function interface(s)",
  "suggestion": "actionable repair instruction for SignatureAgent — what to change in the signature or docstring, and why",
  "target_functions": ["function_name_1", "function_name_2"]
}}

If the failures are NOT caused by signature/docstring issues (e.g. the problem is
in the function body implementations, or outside the generated functions):
{{
  "error_type": "tool_error",
  "reason": "explanation of why the signature/docstring is not at fault",
  "suggestion": "",
  "target_functions": []
}}

Rules:
- `image_path` is ALWAYS automatically injected by the system. It will never
  appear in the tool-call transcript. This is EXPECTED and CORRECT. Do NOT
  flag missing image_path as an issue.

- UNIT CONVERSION IS NOT YOUR CONCERN. The system returns values in meters;
  the caller is responsible for converting to inches/feet/etc. A wrong answer
  that differs only by a unit scale (e.g. 0.34 meters vs 30 inches) is a
  TOOL_ERROR or a caller-side issue — NOT a signature/docstring defect. Do NOT
  suggest adding unit parameters or changing return units.

- IGNORE NUMERICAL MAGNITUDE. If the function returned a value but it was the
  wrong number, that is an implementation bug (tool_error), not a signature error.

- ONLY flag signature_error for FUNCTIONAL INTERFACE DEFECTS:
  * Docstring describes the WRONG COMPUTATION (e.g. docstring says "computes
    horizontal distance" but the function is meant for VERTICAL distance).
  * Signature is missing a MANDATORY input (e.g. a distance function that needs
    two bounding boxes but only accepts one).
  * Wrong parameter TYPE (e.g. boundingbox should be list but is str).
  * Wrong return TYPE (e.g. should return float but returns list).
  * Docstring contradicts the signature (parameter names/types don't match).
  * A needed function is MISSING from the registry entirely.

- NEVER suggest deleting or renaming a function unless you have verified against
  ALL SAMPLED QUESTIONS that no question type depends on it. If you need a
  function for a new purpose, suggest ADDING it, not hijacking an existing one.

- If the signatures and docstrings are functionally correct for their intended
  purpose but the answers are wrong, classify as tool_error.

- Cross-reference: if the same function appears across multiple wrong answers,
  check whether the INTERFACE is flawed or just the implementation.

- Do NOT analyze each case individually. Output ONE conclusion.

{wrong_cases}

Modifiable generated functions (signatures and docstrings — primitives excluded):
{function_registry}
"""
