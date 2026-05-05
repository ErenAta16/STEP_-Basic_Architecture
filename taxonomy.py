"""
Controlled mathematics taxonomy used by the web UI and Layer 1b.

Mirrors the MathE question schema (Topic -> Subtopic -> Keywords) so the
pipeline can expose a consistent vocabulary to the user instead of raw
regex labels. Keep the lists alphabetically sorted within each bucket so
diffs stay readable when new items are added.

The matching rules are intentionally simple (case-insensitive regex
patterns). They are good enough for the short problem statements produced
by Layer 0/3; for anything more ambitious a proper classifier could be
plugged behind the same ``classify_taxonomy`` call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# For each subtopic we store:
#   * display name (what the UI shows)
#   * a list of regex patterns that, when any matches, promote this subtopic
#   * the allowed keyword pool with their own patterns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeywordRule:
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class SubtopicRule:
    name: str
    patterns: tuple[str, ...]
    keywords: tuple[KeywordRule, ...]


@dataclass(frozen=True)
class TopicRule:
    name: str
    subtopics: tuple[SubtopicRule, ...]


def _kw(name: str, *patterns: str) -> KeywordRule:
    return KeywordRule(name=name, patterns=tuple(patterns))


# Shared keyword pools reused across integration subtopics.
_INT_TECHNIQUES: tuple[KeywordRule, ...] = (
    _kw("Direct integrals", r"\bdirect\s+integrals?\b", r"\belementary integral\b"),
    _kw("Fundamental theorem of Calculus",
        r"fundamental\s+theorem", r"\bFTC\b"),
    _kw("Integration by parts",
        r"integration\s+by\s+parts", r"\bby\s+parts\b",
        r"\bu\s*dv\b", r"\\int\s*u\s*dv"),
    _kw("Partial fractions decomposition",
        r"partial\s+fractions?", r"\bdecomposition\b"),
    _kw("Region decomposition",
        r"region\s+decomposition", r"split(ting)?\s+the\s+region"),
    _kw("Substitution",
        r"\bsubstitution\b", r"\bu-?substitution\b", r"\bchange\s+of\s+variable"),
    _kw("Trigonometric functions",
        r"\\sin", r"\\cos", r"\\tan", r"\bsin\b", r"\bcos\b", r"\btan\b"),
    _kw("Trigonometric substitution",
        r"trigonometric\s+substitution", r"trig\s+sub"),
    _kw("Area of a planar region",
        r"area\s+of\s+(a\s+)?(planar\s+)?region", r"bounded\s+region"),
    _kw("Area of a region between two or more curves",
        r"area\s+between\s+(the\s+)?curves", r"between\s+two\s+curves"),
    _kw("Volume of revolution",
        r"volume\s+of\s+revolution", r"solid\s+of\s+revolution"),
    _kw("Power rule",
        r"\bpower\s+rule\b", r"power\s+rule\s+for\s+integration"),
    _kw("Sum rule",
        r"\bsum\s+rule\b", r"linearity\s+of\s+integration"),
    _kw("Logarithmic integration",
        r"\\int\s*1\s*/\s*x", r"\\int\s*\\frac\{1\}\{x\}",
        r"integral\s+of\s+1\s*/\s*x", r"\\ln\s*\|x\|", r"natural\s+log"),
    _kw("Polynomial integration",
        r"polynomial\s+(term|integration|integral)"),
    _kw("Rational functions",
        r"rational\s+function"),
    _kw("Exponential integration",
        r"\\int\s*e\^", r"exponential\s+(function|integration)"),
)


_SURFACE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Parametrization", r"parametri[sz]"),
    _kw("Cross product", r"cross\s+product", r"\\times"),
    _kw("Normal vector", r"normal\s+vector", r"unit\s+normal"),
    _kw("Orientation", r"orientation", r"outward", r"inward"),
    _kw("Spherical coordinates", r"spherical"),
    _kw("Cylindrical coordinates", r"cylindrical"),
    _kw("Flux", r"\bflux\b"),
)


_DIFF_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Chain rule", r"chain\s+rule"),
    _kw("Product rule", r"product\s+rule"),
    _kw("Quotient rule", r"quotient\s+rule"),
    _kw("Implicit differentiation", r"implicit"),
    _kw("Logarithmic differentiation", r"logarithmic\s+differentiation"),
    _kw("Higher-order derivatives",
        r"second\s+derivative", r"higher[-\s]order", r"y''|y\(\s*n\s*\)"),
    _kw("Trigonometric functions", r"\\sin|\\cos|\\tan|\bsin\b|\bcos\b|\btan\b"),
)


_LIMIT_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("L'Hopital's rule", r"l['\u2019]\s*hopital|l['\u2019]\s*h[o\u00f4]pital"),
    _kw("Squeeze theorem", r"squeeze\s+theorem|sandwich\s+theorem"),
    _kw("One-sided limits", r"one[-\s]sided|left[-\s]hand|right[-\s]hand"),
    _kw("Infinity limits", r"\\infty|infinity"),
    _kw("Indeterminate forms",
        r"indeterminate\s+form", r"\b0\s*/\s*0\b", r"\\infty\s*/\s*\\infty"),
)


_SERIES_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Convergence tests", r"convergence\s+test"),
    _kw("Ratio test", r"ratio\s+test"),
    _kw("Root test", r"root\s+test"),
    _kw("Power series", r"power\s+series"),
    _kw("Taylor series", r"\bTaylor\b"),
    _kw("Maclaurin series", r"\bMaclaurin\b"),
    _kw("Geometric series", r"geometric\s+series"),
)


_LA_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Row reduction", r"row\s+reduction|gaussian\s+elimination"),
    _kw("Determinants", r"determinant|\\det"),
    _kw("Eigenvalues", r"eigen\s*value"),
    _kw("Eigenvectors", r"eigen\s*vector"),
    _kw("Matrix rank", r"\brank\b"),
    _kw("Systems of equations", r"system\s+of\s+(linear\s+)?equations"),
)


_ODE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Separable", r"separable"),
    _kw("Integrating factor", r"integrating\s+factor"),
    _kw("Characteristic equation", r"characteristic\s+equation"),
    _kw("Linear ODE", r"linear\s+ode|linear\s+differential"),
    _kw("Non-linear ODE", r"non[-\s]?linear\s+ode"),
)


TAXONOMY: tuple[TopicRule, ...] = (
    TopicRule(
        name="Integration",
        subtopics=(
            SubtopicRule(
                name="Indefinite Integrals",
                patterns=(
                    r"\\int(?!_)(?!\\limits_)",
                    r"indefinite\s+integral",
                    r"antiderivative",
                    r"\+\s*C\b",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Definite Integrals",
                patterns=(
                    r"\\int_",
                    r"\\int\\limits_",
                    r"definite\s+integral",
                    r"evaluate\s+the\s+integral",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Improper Integrals",
                patterns=(
                    r"improper\s+integral",
                    r"\\int[^\n]*\\infty",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Double Integrals",
                patterns=(r"\\iint", r"double\s+integral"),
                keywords=_INT_TECHNIQUES + (
                    _kw("Polar coordinates", r"polar\s+coordinate"),
                    _kw("Change of order", r"change\s+(the\s+)?order"),
                ),
            ),
            SubtopicRule(
                name="Triple Integrals",
                patterns=(r"\\iiint", r"triple\s+integral"),
                keywords=_INT_TECHNIQUES + (
                    _kw("Cylindrical coordinates", r"cylindrical"),
                    _kw("Spherical coordinates", r"spherical"),
                ),
            ),
            SubtopicRule(
                name="Surface Integrals",
                patterns=(
                    r"surface\s+integral",
                    r"\\iint[^\n]*dS",
                    r"\bdS\b",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Line Integrals",
                patterns=(
                    r"line\s+integral",
                    r"\\oint",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Differentiation",
        subtopics=(
            SubtopicRule(
                name="Derivatives",
                patterns=(
                    r"\\frac\{d",
                    r"\bderivative\b",
                    r"differentiate",
                    r"\bdy\s*/\s*dx\b",
                ),
                keywords=_DIFF_KEYWORDS,
            ),
            SubtopicRule(
                name="Partial Derivatives",
                patterns=(
                    r"\\frac\{\\partial",
                    r"partial\s+derivative",
                ),
                keywords=_DIFF_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Limits and Continuity",
        subtopics=(
            SubtopicRule(
                name="Limits",
                patterns=(r"\\lim\b", r"\blimit\b"),
                keywords=_LIMIT_KEYWORDS,
            ),
            SubtopicRule(
                name="Continuity",
                patterns=(r"continuous", r"continuity"),
                keywords=_LIMIT_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Series",
        subtopics=(
            SubtopicRule(
                name="Numerical Series",
                patterns=(r"\\sum", r"\bseries\b"),
                keywords=_SERIES_KEYWORDS,
            ),
            SubtopicRule(
                name="Power Series",
                patterns=(r"power\s+series",),
                keywords=_SERIES_KEYWORDS,
            ),
            SubtopicRule(
                name="Taylor/Maclaurin",
                patterns=(r"\bTaylor\b", r"\bMaclaurin\b"),
                keywords=_SERIES_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Differential Equations",
        subtopics=(
            SubtopicRule(
                name="First-order ODEs",
                patterns=(
                    r"first[-\s]order",
                    r"\by'\b(?!\')",
                    r"dy\s*/\s*dx",
                ),
                keywords=_ODE_KEYWORDS,
            ),
            SubtopicRule(
                name="Higher-order ODEs",
                patterns=(
                    r"second[-\s]order",
                    r"y''",
                    r"higher[-\s]order\s+ode",
                ),
                keywords=_ODE_KEYWORDS,
            ),
            SubtopicRule(
                name="Partial Differential Equations",
                patterns=(r"\bPDE\b", r"partial\s+differential\s+equation"),
                keywords=_ODE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Linear Algebra",
        subtopics=(
            SubtopicRule(
                name="Matrices",
                patterns=(r"\bmatrix\b", r"\bmatrices\b"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Determinants",
                patterns=(r"determinant", r"\\det"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Eigenvalues and Eigenvectors",
                patterns=(r"eigen\s*value", r"eigen\s*vector"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Systems of Equations",
                patterns=(r"system\s+of\s+(linear\s+)?equations",),
                keywords=_LA_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Vector Calculus",
        subtopics=(
            SubtopicRule(
                name="Divergence Theorem",
                patterns=(
                    r"divergence\s+theorem",
                    r"\\nabla\s*\\cdot",
                    r"\bGauss\b",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Stokes' Theorem",
                patterns=(r"Stokes", r"\\nabla\s*\\times", r"\bcurl\b"),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Flux",
                patterns=(r"\bflux\b", r"F\s*\\cdot\s*dS"),
                keywords=_SURFACE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Algebra",
        subtopics=(
            SubtopicRule(
                name="Equation Solving",
                patterns=(
                    r"solve\s+the\s+equation",
                    r"find\s+(?:the\s+)?roots",
                    r"\bquadratic\b",
                ),
                keywords=(
                    _kw("Quadratic formula", r"quadratic\s+formula"),
                    _kw("Factoring", r"\bfactor(?:ing|ise|ize)?\b"),
                    _kw("Rational roots", r"rational\s+root"),
                ),
            ),
            SubtopicRule(
                name="Inequalities",
                patterns=(r"inequalit(?:y|ies)",),
                keywords=(
                    _kw("Sign analysis", r"sign\s+analysis"),
                    _kw("Interval notation", r"interval\s+notation"),
                ),
            ),
        ),
    ),
)


def _count_hits(patterns: tuple[str, ...], text: str) -> int:
    """Return how many of ``patterns`` match somewhere in ``text`` (case-insensitive)."""
    if not text:
        return 0
    hits = 0
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                hits += 1
        except re.error:
            continue
    return hits


def _best_subtopic(text: str) -> tuple[TopicRule, SubtopicRule, int] | None:
    """Find the (topic, subtopic) pair with the highest number of pattern hits."""
    best: tuple[TopicRule, SubtopicRule, int] | None = None
    for topic in TAXONOMY:
        for sub in topic.subtopics:
            hits = _count_hits(sub.patterns, text)
            if hits == 0:
                continue
            if best is None or hits > best[2]:
                best = (topic, sub, hits)
    return best


def classify_taxonomy(text: str, max_keywords: int = 5) -> dict:
    """Return ``{topic, subtopic, keywords}`` for the given problem text.

    ``keywords`` is capped at ``max_keywords`` entries (minimum two when the
    subtopic's pool provides enough matches). When no subtopic pattern fires
    ``topic`` and ``subtopic`` are empty strings, but the function still runs
    a keyword scan against the union of all known keyword rules so that the
    UI always shows something useful when possible.
    """
    text = text or ""
    picked = _best_subtopic(text)

    if picked is None:
        pool: list[KeywordRule] = []
        for topic in TAXONOMY:
            for sub in topic.subtopics:
                pool.extend(sub.keywords)
        seen: set[str] = set()
        unique_pool: list[KeywordRule] = []
        for rule in pool:
            if rule.name in seen:
                continue
            seen.add(rule.name)
            unique_pool.append(rule)
        keywords = [r.name for r in unique_pool if _count_hits(r.patterns, text)]
        return {
            "topic": "",
            "subtopic": "",
            "keywords": keywords[:max_keywords],
        }

    topic, sub, _ = picked
    keywords: list[tuple[str, int]] = []
    for rule in sub.keywords:
        hits = _count_hits(rule.patterns, text)
        if hits:
            keywords.append((rule.name, hits))

    keywords.sort(key=lambda kv: (-kv[1], kv[0].lower()))
    keyword_names = [name for name, _ in keywords[:max_keywords]]

    return {
        "topic": topic.name,
        "subtopic": sub.name,
        "keywords": keyword_names,
    }


def _find_subtopic(topic_name: str, subtopic_name: str) -> SubtopicRule | None:
    """Locate a subtopic rule by display name (falls back to ``None``)."""
    tn = (topic_name or "").strip().lower()
    sn = (subtopic_name or "").strip().lower()
    if not tn or not sn:
        return None
    for topic in TAXONOMY:
        if topic.name.lower() != tn:
            continue
        for sub in topic.subtopics:
            if sub.name.lower() == sn:
                return sub
    return None


def keywords_for_subtopic(topic_name: str, subtopic_name: str, text: str) -> list[str]:
    """Return keyword names from the given subtopic pool that match ``text``.

    Keeps the result ordered by pattern-hit count (descending) and falls back
    to case-insensitive name ordering on ties.
    """
    sub = _find_subtopic(topic_name, subtopic_name)
    if sub is None or not text:
        return []
    scored: list[tuple[str, int]] = []
    for rule in sub.keywords:
        hits = _count_hits(rule.patterns, text)
        if hits:
            scored.append((rule.name, hits))
    scored.sort(key=lambda kv: (-kv[1], kv[0].lower()))
    return [name for name, _ in scored]


def merge_keywords(*lists: list[str]) -> list[str]:
    """Concatenate keyword lists preserving first-seen order and removing duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for items in lists:
        for name in items or []:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def topic_from_keywords(keywords: list[str]) -> tuple[str, str] | None:
    """Infer ``(topic, subtopic)`` from a list of already-chosen keyword names.

    Finds the subtopic whose keyword pool best matches the given list (by
    count of matches). Useful as a fallback for sources whose text does not
    trigger the regex-based classifier — e.g. an English video summary that
    never uses the ``\\int`` LaTeX macro. Returns ``None`` when nothing matches.
    """
    if not keywords:
        return None

    wanted = {k.strip().lower() for k in keywords if k and k.strip()}
    if not wanted:
        return None

    best: tuple[int, TopicRule, SubtopicRule] | None = None
    for topic in TAXONOMY:
        for sub in topic.subtopics:
            pool_lower = {r.name.lower() for r in sub.keywords}
            hits = len(wanted & pool_lower)
            if hits == 0:
                continue
            if best is None or hits > best[0]:
                best = (hits, topic, sub)

    if best is None:
        return None
    _, topic, sub = best
    return (topic.name, sub.name)
