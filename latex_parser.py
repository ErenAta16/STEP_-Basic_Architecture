"""
LaTeX-ish fragments → SymPy: used by L6 to compare LLM answers to references.
"""

import re
from sympy import sympify


def find_matching_brace(s: str, pos: int) -> int:
    """Return the index of the closing `}` that pairs with `s[pos]`=`{`."""
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
    """Strip `$`, expand common macros, then hand off to `sympify`."""
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

    # ^{...} -> **(...) us notasyonu
    while "^{" in s:
        idx = s.index("^{")
        brace_start = idx + 1
        brace_end = find_matching_brace(s, brace_start)
        content = s[brace_start + 1:brace_end]
        content = latex_to_sympy(content)
        s = s[:idx] + f"**({content})" + s[brace_end + 1:]

    # ^N (single-digit exponent) -> **N
    s = re.sub(r'\^(\d)', r'**\1', s)

    # Temel donusumler
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

    # Implicit multiplication (bitisik karakterler)
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
    """SymPy evalf → Python float, or None if parsing fails."""
    try:
        parsed = latex_to_sympy(latex_str)
        expr = sympify(parsed, locals={"pi": __import__("sympy").pi,
                                        "sqrt": __import__("sympy").sqrt,
                                        "E": __import__("sympy").E,
                                        "a": __import__("sympy").Symbol("a", positive=True)})
        return float(expr.evalf())
    except Exception:
        return None


def parse_latex_to_expr(latex_str: str):
    """Return a SymPy object (may still contain symbols)."""
    try:
        parsed = latex_to_sympy(latex_str)
        from sympy import pi as sym_pi, sqrt as sym_sqrt, E as sym_E, Symbol
        expr = sympify(parsed, locals={"pi": sym_pi, "sqrt": sym_sqrt,
                                        "E": sym_E,
                                        "a": Symbol("a", positive=True)})
        return expr
    except Exception:
        return None


if __name__ == "__main__":
    from sympy import pi, sqrt

    test_cases = [
        (r"$\frac{4\pi}{3}$", 4 * pi / 3),
        (r"$\frac{\sqrt{21}}{3}$", sqrt(21) / 3),
        (r"$\frac{4(9\sqrt{3} + 4\sqrt{2} - 2)}{105}$", 4 * (9 * sqrt(3) + 4 * sqrt(2) - 2) / 105),
        (r"$\frac{364\pi\sqrt{2}}{3}$", 364 * pi * sqrt(2) / 3),
        (r"$\frac{13\sqrt{2}}{12}$", 13 * sqrt(2) / 12),
    ]

    print("LaTeX parser smoke test\n")
    ok = 0
    for latex_str, expected in test_cases:
        parsed = latex_to_sympy(latex_str)
        val = parse_latex_to_value(latex_str)
        expected_val = float(expected.evalf())

        if val is not None and abs(val - expected_val) < 0.01:
            print(f"  [OK] {latex_str}")
            print(f"       Parsed: {parsed} -> {val:.6f} (expected {expected_val:.6f})")
            ok += 1
        else:
            print(f"  [FAIL] {latex_str}")
            print(f"         Parsed: {parsed}, value: {val}")

    print(f"\nResult: {ok}/{len(test_cases)}")
