"""
Convert small LaTeX fragments to SymPy for numeric comparison in ``STEPSolver._solve_with_consensus``.

Not a full LaTeX parser—only patterns common in model ``\\boxed{}`` / final-line output.
Uses a shared ``_SYM_LOCALS`` dict for ``sympify`` (``pi``, ``sqrt``, ``E``, positive ``a``).
"""

import re
from sympy import E, Symbol, pi, sqrt, sympify

# Shared namespace for ``sympify`` (avoids repeated ``__import__("sympy")`` per call).
_SYM_LOCALS = {"pi": pi, "sqrt": sqrt, "E": E, "a": Symbol("a", positive=True)}


def find_matching_brace(s: str, pos: int) -> int:
    """Index of the ``}`` that closes the ``{`` at ``s[pos]`` (``pos`` must point at ``{``)."""
    depth = 1
    i = pos + 1
    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    return i - 1


def latex_to_sympy(s: str) -> str:
    """Normalize dollars and common macros into a string ``sympify`` can read."""
    s = s.replace("$", "").strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # \frac{...}{...}
    while "\\frac" in s:
        idx = s.index("\\frac")
        num_start = s.index("{", idx)
        num_end = find_matching_brace(s, num_start)
        numerator = s[num_start + 1:num_end]
        den_start = s.index("{", num_end + 1)
        den_end = find_matching_brace(s, den_start)
        denominator = s[den_start + 1:den_end]
        numerator = latex_to_sympy(numerator)
        denominator = latex_to_sympy(denominator)
        s = s[:idx] + f"(({numerator})/({denominator}))" + s[den_end + 1:]

    # \sqrt{...}
    while "\\sqrt{" in s:
        idx = s.index("\\sqrt{")
        brace_start = idx + 5
        brace_end = find_matching_brace(s, brace_start)
        content = s[brace_start + 1:brace_end]
        content = latex_to_sympy(content)
        s = s[:idx] + f"sqrt({content})" + s[brace_end + 1:]

    # \mathrm{e} -> E (Euler number), \mathrm{...} -> ...
    s = re.sub(r'\\mathrm\{e\}', 'E', s)
    s = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', s)
    s = re.sub(r'\\text\{e\}', 'E', s)
    s = re.sub(r'\\text\{([^}]+)\}', r'\1', s)
    s = s.replace("\\ln", "log")
    # Standalone e -> E (Euler's number) in final answer context
    s = re.sub(r'(?<![a-df-zA-DF-Z])e(?![a-df-zA-DF-Z])', 'E', s)

    # ^{...} -> **(...)  (Python power syntax for sympify)
    while "^{" in s:
        idx = s.index("^{")
        brace_start = idx + 1
        brace_end = find_matching_brace(s, brace_start)
        content = s[brace_start + 1:brace_end]
        content = latex_to_sympy(content)
        s = s[:idx] + f"**({content})" + s[brace_end + 1:]

    # ^N (single-digit exponent) -> **N
    s = re.sub(r'\^(\d)', r'**\1', s)

    # Common LaTeX spacing and operators
    s = s.replace("\\pi", "(pi)")
    s = s.replace("\\cdot", "*")
    s = s.replace("\\times", "*")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\!", "")
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left", "").replace("\\right", "")

    # Spaces → implicit multiplication (after cleanup above)
    # "(pi) a" -> "(pi)*a",  ") (" -> ")*("
    s = re.sub(r'\)\s+\(', ')*(', s)
    s = re.sub(r'\)\s+(\w)', r')*\1', s)
    s = re.sub(r'(\w)\s+\(', r'\1*(', s)
    s = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z])', r'\1*\2', s)

    # Implicit multiplication (adjacent factors)
    s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)         # 4a -> 4*a, 4( -> 4*(
    s = re.sub(r'(\))(\()', r'\1*\2', s)                  # )( -> )*(
    s = re.sub(r'(\))([\da-zA-Z])', r'\1*\2', s)         # )2 -> )*2, )a -> )*a
    s = re.sub(r'([a-zA-Z])\*\(', lambda m:
        m.group(0) if m.group(1) in ('sqrt', 'log', 'sin', 'cos', 'tan', 'exp')
        else m.group(1) + '*(', s)

    # Undo over-eager `*` before `(` for known function names
    for fn in ('sqrt', 'log', 'sin', 'cos', 'tan', 'exp', 'pi'):
        s = s.replace(f'{fn}*(', f'{fn}(')

    return s


def parse_latex_to_value(latex_str: str) -> float | None:
    """Evaluate to a float, or ``None`` if the fragment is not numeric enough."""
    try:
        parsed = latex_to_sympy(latex_str)
        expr = sympify(parsed, locals=_SYM_LOCALS)
        return float(expr.evalf())
    except Exception:
        return None


def parse_latex_to_expr(latex_str: str):
    """Like ``parse_latex_to_value`` but keeps a SymPy expression (e.g. symbolic ``a``)."""
    try:
        parsed = latex_to_sympy(latex_str)
        expr = sympify(parsed, locals=_SYM_LOCALS)
        return expr
    except Exception:
        return None
