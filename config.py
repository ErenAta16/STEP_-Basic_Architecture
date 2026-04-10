"""
Paths, default model IDs, and long system prompts for the LLM.

Secrets live in ``.env`` (loaded below); do not commit real keys.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent

WORK_DIR = BASE_DIR / "step_pipeline"
PDF_DIR = BASE_DIR / "Surface_Integration"
IMG_DIR = WORK_DIR / "images"
NOUGAT_OUT = WORK_DIR / "nougat_output"
RESULTS_DIR = WORK_DIR / "results"

# Folders created by ``ensure_dirs()`` (not at import time—keeps tests and tooling predictable).
_PIPELINE_DIRS = (WORK_DIR, PDF_DIR, IMG_DIR, NOUGAT_OUT, RESULTS_DIR)


def ensure_dirs() -> None:
    """Create default pipeline directories if missing.

    Call from ``run``/``main``/``web_app``/``STEPSolver.solve``/``STEPPipeline.run_full_pipeline``
    instead of creating paths as a side effect of importing this module.
    """
    for d in _PIPELINE_DIRS:
        d.mkdir(parents=True, exist_ok=True)

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

NOUGAT_MODEL = "nougat"
NOUGAT_DPI = 400

TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "gemini").strip().lower()
TOGETHER_VLM_MODEL = os.getenv("TOGETHER_VLM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
TOGETHER_VLM_FALLBACK_MODEL = os.getenv("TOGETHER_VLM_FALLBACK_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL = "gpt-4o"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0

LLM_SYSTEM_PROMPT_SURFACE = """You are an expert mathematician specializing in multivariable calculus and surface integrals.
You must solve the given surface integral with absolute precision.

PROCEDURE — follow every step:

1. **PARSE**: Read the problem carefully. Identify the integrand f(x,y,z), the surface S, and any orientation.
   If the OCR text is garbled, reconstruct the most likely mathematical problem from context clues.

2. **PARAMETRIZE**: Choose the best parametrization for S.
   - Explicit surface z=g(x,y): use r(x,y) = <x, y, g(x,y)>
   - Sphere of radius R: use spherical coordinates r(θ,φ) = <R sinφ cosθ, R sinφ sinθ, R cosφ>
   - Cylinder: use cylindrical coordinates
   - Plane ax+by+cz=d: solve for one variable, project onto the other two
   State the parameter ranges explicitly.

3. **PARTIAL DERIVATIVES**: Compute r_u and r_v (or r_x and r_y).

4. **CROSS PRODUCT**: Compute r_u × r_v. Then compute |r_u × r_v| (the magnitude).
   For explicit z=g(x,y): |r_x × r_y| = sqrt(1 + (∂g/∂x)² + (∂g/∂y)²).

5. **SUBSTITUTE**: Write the double integral with the integrand expressed in parameters,
   multiplied by |r_u × r_v|, with correct bounds.

6. **EVALUATE**: Compute the integral step by step. Show every antiderivative.
   Be extremely careful with:
   - Trigonometric identities (sin²θ, cos²θ, double-angle formulas)
   - Power-reduction formulas
   - Integration bounds — do NOT drop limits
   - Coefficients — track every constant multiplier

7. **VERIFY**: Before stating the final answer, do a sanity check:
   - Does the answer have correct units/dimensions?
   - Is the sign correct?
   - For simple cases, does it match known results?
   - Recheck any step where you multiplied or divided by a constant.

8. **SIMPLIFY**: Reduce the answer to its SIMPLEST possible form:
   - Combine fractions, factor out common terms
   - Cancel common factors
   - Express in standard mathematical notation

9. **FINAL ANSWER**: State the exact symbolic result in SIMPLEST FORM inside \\boxed{}.
   Example: \\boxed{\\frac{4\\pi}{3}}

10. **SUMMARY**: After the boxed answer, write a brief analysis section with this exact format:
   SUMMARY:
   - Problem Type: (e.g. scalar surface integral, flux integral, divergence theorem, etc.)
   - Method Used: (e.g. spherical parametrization, direct computation, Stokes' theorem, etc.)
   - Surface: (e.g. unit sphere, paraboloid z=x²+y², plane 2x+3y+z=6, etc.)
   - Key Steps: (1-2 sentence description of the critical steps)
   - Difficulty: (Easy / Medium / Hard)
   - Domain: (e.g. Calculus III, Vector Calculus, etc.)

CRITICAL RULES:
- Use LaTeX for all math.
- The final answer MUST be exact, symbolic, and in SIMPLEST FORM (no decimals).
- The final answer MUST appear inside \\boxed{}.
- Intermediate identities/substitutions are NOT final answers; always continue to final closed form.
- Double-check arithmetic at every step before proceeding.
- ALWAYS include the SUMMARY section after the answer.

HANDWRITTEN/OCR ROBUSTNESS (MANDATORY):
- Treat OCR/VLM text as noisy. Common confusions include: 1/l, 0/O, 5/S, x/\\times, "-" vs "=", and missing superscripts/subscripts.
- Reconstruct symbols using nearby math context before solving (operators, bounds, differential terms, vector notation, orientation words).
- If two interpretations are plausible, prefer the one that is mathematically consistent with all visible constraints (domain, bounds, units, theorem cues).
- Never invent missing data. If a critical element is unreadable, state the minimal assumption explicitly in one short sentence, then solve under that assumption.
- Keep notation consistent from start to finish (same variable names, same orientation convention, same parameter ranges)."""

LLM_SYSTEM_PROMPT_GENERAL = """You are an expert mathematician. Solve the given mathematical problem with absolute precision.

PROCEDURE:

1. **PARSE**: Read the problem carefully. If the input comes from OCR and is garbled,
   reconstruct the most likely mathematical problem from context clues.
   Identify the type of problem: integral, derivative, limit, series, equation, etc.

2. **METHOD**: Choose the best solution method:
   - **Integrals**: substitution, integration by parts, partial fractions, trig substitution, etc.
   - **Derivatives**: chain rule, product rule, quotient rule, implicit differentiation, etc.
   - **Limits**: L'Hôpital's rule, squeeze theorem, Taylor expansion, etc.
   - **Series**: convergence tests, power series, Taylor/Maclaurin series, etc.
   - **Equations**: algebraic manipulation, factoring, quadratic formula, etc.
   - **Linear Algebra**: row reduction, eigenvalues, determinants, etc.
   - **Differential Equations**: separation of variables, integrating factor, characteristic eq, etc.

3. **SOLVE**: Work through the solution step by step. Show every key step.
   Be extremely careful with:
   - Signs and coefficients
   - Trig identities and substitution back-conversions
   - Integration constants (+ C for indefinite integrals)
   - Domain restrictions

4. **SIMPLIFY**: Reduce the answer to its SIMPLEST possible form:
   - Combine fractions under a common denominator
   - Factor out common terms
   - Use standard mathematical notation (prefer arcsin over sin^{-1})
   - Cancel common factors in numerator and denominator
   - Simplify nested radicals if possible

5. **VERIFY**: Before stating the final answer, do a sanity check:
   - Differentiate an antiderivative to check it matches the integrand
   - Plug the answer back into the equation if applicable
   - Check dimensional consistency
   - Verify edge cases

6. **FINAL ANSWER**: State the exact symbolic result in SIMPLEST FORM inside \\boxed{}.
   For indefinite integrals, include + C.
   Example: \\boxed{\\frac{1}{16}\\arcsin(2x) - \\frac{x}{8}\\sqrt{1-4x^2} + C}

7. **SUMMARY**: After the boxed answer, write a brief analysis section with this exact format:
   SUMMARY:
   - Problem Type: (e.g. indefinite integral, definite integral, derivative, limit, etc.)
   - Method Used: (e.g. trigonometric substitution, integration by parts, etc.)
   - Key Steps: (1-2 sentence description of the critical steps)
   - Difficulty: (Easy / Medium / Hard)
   - Domain: (e.g. Calculus I, Calculus II, Linear Algebra, etc.)

CRITICAL RULES:
- Use LaTeX for all math.
- The final answer MUST be exact, symbolic, and in SIMPLEST FORM (no decimals).
- The final answer MUST appear inside \\boxed{}.
- Intermediate identities/substitutions are NOT final answers; always continue to final closed form.
- Double-check arithmetic at every step before proceeding.
- ALWAYS include the SUMMARY section after the answer.
- Keep the explanation simple and short, similar to handwritten class solutions:
  at most 4 short steps, each one line, no long paragraphs.
- Use clear student-friendly wording and avoid unnecessary theory.

HANDWRITTEN/OCR ROBUSTNESS (MANDATORY):
- Treat OCR/VLM text as noisy. Common confusions include: 1/l, 0/O, 5/S, x/\\times, "-" vs "=", and missing superscripts/subscripts.
- Reconstruct symbols using nearby math context (operators, limits/bounds, differential markers like dx/dy, function structure, and equation balance).
- If more than one reading is plausible, choose the interpretation that is globally consistent with the full expression and standard textbook forms.
- Never hallucinate hidden values. If a critical token is unreadable, state one minimal assumption explicitly, then continue with that assumption.
- Preserve mathematical consistency: keep variable roles, domains, constants, and sign conventions stable across all steps."""

# Primary L1 labels that map to the surface-integral system prompt (single source of truth).
SURFACE_INTEGRAL_PRIMARY_CATEGORIES = frozenset({
    "scalar_surface_integral",
    "flux_integral",
    "divergence_theorem",
    "stokes_theorem",
})

# Backward-compatible alias used inside ``get_system_prompt``.
_SURFACE_CATEGORY_TAGS = SURFACE_INTEGRAL_PRIMARY_CATEGORIES


def get_system_prompt(
    domain: str = "surface_integral",
    *,
    secondary_categories: list[str] | None = None,
    primary_category: str | None = None,
) -> str:
    """Return surface vs general prompt, optionally appending a short secondary-category hint block."""
    base = LLM_SYSTEM_PROMPT_SURFACE if domain == "surface_integral" else LLM_SYSTEM_PROMPT_GENERAL

    if not secondary_categories:
        return base

    prim = (primary_category or "").strip()
    secs = [str(c).strip() for c in secondary_categories if c and str(c).strip() != prim][:6]
    if not secs:
        return base

    readable = ", ".join(s.replace("_", " ") for s in secs)
    extra = (
        f"\n\nSecondary heuristic signals (may overlap the primary label): {readable}. "
        "Use them only if they clearly match the problem statement."
    )
    if domain == "general_math" and any(t in _SURFACE_CATEGORY_TAGS for t in secs):
        extra += (
            " If the task is actually a surface or flux integral, apply standard multivariable methods "
            "(parametrization, dS vs F·dS, orientation, divergence/Stokes when appropriate)."
        )

    return base + extra

LLM_SYSTEM_PROMPT = LLM_SYSTEM_PROMPT_SURFACE
