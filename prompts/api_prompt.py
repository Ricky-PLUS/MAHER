API_PROMPT = """
You are an expert at implementing Python functions based on a given docstring and signature.
Your implementation MUST call ONLY the functions listed in PRIMITIVES and CURRENT API below.
You may use Python standard library modules (e.g. `import math`) when needed.

===== PRIMITIVES (foundational building blocks — read‑only) =====
{primitives}

===== CURRENT API (generated higher‑level functions) =====
{predef_signatures}
{generated_signatures}

===== RETRIEVED TOOLS =====
{rag_tools}

===== OUTPUT FORMAT =====
Output ONLY the function body inside <implementation></implementation> tags.
Do NOT include the "def" line or ```python``` tags.

Here is an example — note how stdlib imports go at the top, and lines inside 'if' are indented 4 spaces deeper:

<docstring>
\"\"\"
Computes the hypotenuse of two numbers and returns it as JSON.

Args:
    a (float): First number.
    b (float): Second number.

Returns:
    str: A JSON string with the result.
\"\"\"
</docstring>
<signature>def _hypotenuse_json(a: float, b: float) -> str:</signature>
<implementation>
import json
import math
c = math.sqrt(a * a + b * b)
if c < 0:
    return json.dumps(0.0)
return json.dumps(c)
</implementation>

===== RULES =====
1. Only call functions explicitly listed in the PRIMITIVES or CURRENT API sections above.
2. Always check for empty/None results from API calls before using the returned values.
3. DO NOT round answers. Keep full float precision.
4. If a critic report is provided, fix the concrete implementation issue it names.
5. If you need Python standard library modules (e.g. `math`, `json`, `os`), include the `import` statement at the top of your implementation body.
6. CRITICAL: Only output the implementation body. Do NOT rename the function, change parameters, change the return type, or modify the docstring or signature in any way.

Critic report, when available:
{critic_report}

Now implement the following function:

<docstring>
{docstring}
</docstring>
<signature>
{signature}
</signature>
"""
