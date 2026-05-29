SIGNATURE_PROMPT = """
You are designing new higher-level function signatures to extend an API.
All new functions MUST be built exclusively on top of the PRIMITIVES listed below.
DO NOT invent new low-level operations — compose from what already exists.

===== PRIMITIVES (foundational building blocks — read‑only) =====
{primitives}

===== CURRENT API (generated higher‑level functions) =====
{signatures}

===== RETRIEVED TOOLS FROM KNOWLEDGE BASE =====
{rag_tools}

===== TASK =====
I will show you a series of questions. Your job is to propose NEW function signatures
(with docstrings) that would help answer these questions by composing the existing
primitives and tools.


===== DESIGN RULES =====
1. New methods MUST start with an underscore (e.g. _get_result).
2. Parameters should use standard Python types: str, list, float, int.
3. Every new function MUST internally call at least one primitive from the available API.
4. Functions should be simple single-purpose wrappers.
5. CRITICAL — Before proposing ANY function, scan the CURRENT API and RETRIEVED TOOLS above
   and verify the function does NOT already exist. If a function with the same purpose
   (even if named differently) is already available, do NOT propose it.
6. Only add new methods when truly needed. If the problem can be solved with existing
   combinations, do not add anything.
7. The <purpose> tag MUST contain a single concise sentence (no more than one line)
   describing the function's core functionality AND the meaning of its return value
   (e.g., what each possible return value represents). It will be used as the tool
   description when the function is exposed to an agent for tool-calling.

===== OUTPUT EXAMPLES (format reference only — these do NOT exist yet) =====
<purpose>
[One concise sentence describing the function's core functionality and the meaning of its return value (e.g., "Returns 0 if the first object is closer, 1 if the second is closer, -1 if equal"). Used as the tool description for agent tool-calling.]
</purpose>
<docstring>
Returns the [description of result].
Args:
    [param_name] ([type]): [Description of parameter].
    [additional args as needed]
Returns:
    [type]: [Description of returned value].
</docstring>
<signature>def _get_[descriptive_name]([params]) -> [return_type]:</signature>


Ground truth hints, when available:
{ground_truth}

Critic report, when available:
{critic_report}

Here is the question:
{question}
"""

