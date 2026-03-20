"""
Layer 1 — cheap text profiling: regex/heuristics → keywords, category, surface hints.
Keyword lists mix EN/TR spellings so multilingual PDFs still match.
"""

import re


PROBLEM_CATEGORIES = {
    # --- Surface integral types ---
    "scalar_surface_integral": [
        r"∬.*dS", r"\\iint.*dS", r"surface integral",
        r"find the area", r"evaluate.*surface",
    ],
    "flux_integral": [
        r"flux", r"F\s*[·⋅]\s*dS", r"F\s*\\cdot\s*dS",
        r"across.*surface", r"through.*surface",
    ],
    "divergence_theorem": [
        r"divergence", r"Gauss", r"\\nabla\s*\\cdot",
        r"div\s*F", r"outward flux.*closed",
    ],
    "stokes_theorem": [
        r"Stokes", r"curl", r"\\nabla\s*\\times",
        r"line integral.*surface", r"circulation",
    ],
    # --- General math types ---
    "indefinite_integral": [
        r"\\int(?!\w).*dx", r"∫.*dx", r"∫.*dy", r"∫.*dt", r"∫.*du",
        r"antiderivative", r"belirsiz integral",
    ],
    "definite_integral": [
        r"\\int_", r"∫_", r"\\int\\limits",
        r"evaluate.*integral", r"belirli integral",
    ],
    "double_integral": [
        r"\\iint", r"∬", r"double integral", r"çift integral",
    ],
    "triple_integral": [
        r"\\iiint", r"∭", r"triple integral", r"üçlü integral",
    ],
    "derivative": [
        r"\\frac\{d", r"\\frac\{\\partial", r"derivative",
        r"differentiate", r"türev", r"d/dx",
    ],
    "limit": [
        r"\\lim", r"limit", r"→", r"\\to", r"\\rightarrow",
    ],
    "series": [
        r"\\sum", r"∑", r"series", r"convergence",
        r"Taylor", r"Maclaurin", r"power series", r"seri",
    ],
    "differential_equation": [
        r"differential equation", r"ODE", r"PDE",
        r"y'", r"y''", r"diferansiyel denklem",
    ],
    "linear_algebra": [
        r"matrix", r"matris", r"eigenvalue", r"determinant",
        r"\\det", r"rank", r"eigenvector",
    ],
    "equation": [
        r"solve.*equation", r"find.*x", r"çöz",
        r"roots", r"kök", r"denklem",
    ],
}

SURFACE_TYPES = {
    "sphere": [r"sphere", r"x\^?2\s*\+\s*y\^?2\s*\+\s*z\^?2\s*=", r"kure"],
    "paraboloid": [r"paraboloid", r"z\s*=\s*x\^?2\s*\+\s*y\^?2", r"z\s*=\s*[0-9].*-.*x\^?2"],
    "cylinder": [r"cylinder", r"x\^?2\s*\+\s*y\^?2\s*=", r"silindir"],
    "cone": [r"cone", r"z\s*=\s*\\?sqrt", r"koni"],
    "plane": [r"plane", r"\d+x\s*\+\s*\d+y\s*\+\s*\d+z\s*=", r"duzlem"],
    "hemisphere": [r"hemisphere", r"upper half", r"ust yarikure"],
    "torus": [r"torus"],
}

MATH_KEYWORDS_POOL = [
    ("Surface Integral", r"surface integral|∬|\\iint"),
    ("Flux", r"flux|akı"),
    ("Divergence Theorem", r"divergence|Gauss"),
    ("Stokes Theorem", r"Stokes|curl"),
    ("Parametrization", r"parametri[zs]"),
    ("Cross Product", r"cross product|×|\\times"),
    ("Normal Vector", r"normal vector|unit normal"),
    ("Partial Derivatives", r"partial|∂|\\partial"),
    ("Double Integral", r"double integral|\\iint"),
    ("Triple Integral", r"triple integral|\\iiint"),
    ("Polar Coordinates", r"polar|r\s*dr\s*d.*θ"),
    ("Spherical Coordinates", r"spherical|\\phi.*\\theta"),
    ("Cylindrical Coordinates", r"cylindrical"),
    ("Vector Field", r"vector field|F\s*=|F\s*\("),
    ("Scalar Field", r"scalar field|f\s*\(x"),
    ("Orientation", r"orientation|outward|inward"),
    ("Closed Surface", r"closed surface|kapalı yüzey"),
]


class Layer1_Profiler:
    """Regex-heavy tagging; `latex_text` helps when native PDF text is empty."""

    def profile(self, fname: str, metadata: dict, raw_text: str,
                latex_text: str = "") -> dict:
        combined = raw_text + "\n" + latex_text
        combined_lower = combined.lower()

        keywords = self._extract_keywords(combined)
        category = self._classify_problem(combined_lower)
        surface = self._detect_surface_type(combined_lower)
        summary = self._generate_summary(fname, category, surface, keywords, metadata)

        domain = self._get_domain(category)

        return {
            "keywords": keywords,
            "category": category,
            "domain": domain,
            "surface_type": surface,
            "summary": summary,
            "pages_count": metadata.get("pages", 0),
            "file_size_bytes": int(metadata.get("file_size_kb", 0) * 1024),
            "author": metadata.get("author", ""),
            "producer": metadata.get("producer", ""),
            "creator": metadata.get("creator", ""),
            "creation_date": metadata.get("creation_date", ""),
            "modification_date": metadata.get("modification_date", ""),
        }

    SURFACE_CATEGORIES = {
        "scalar_surface_integral", "flux_integral",
        "divergence_theorem", "stokes_theorem",
    }

    def _get_domain(self, category: str) -> str:
        if category in self.SURFACE_CATEGORIES:
            return "surface_integral"
        if category == "unknown":
            return "general_math"
        return "general_math"

    def _extract_keywords(self, text: str) -> list[str]:
        keywords = []
        for kw, pattern in MATH_KEYWORDS_POOL:
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(kw)
        return keywords

    def _classify_problem(self, text_lower: str) -> str:
        scores = {}
        for cat, patterns in PROBLEM_CATEGORIES.items():
            score = sum(1 for p in patterns if re.search(p, text_lower))
            if score > 0:
                scores[cat] = score

        if not scores:
            return "unknown"
        return max(scores, key=scores.get)

    def _detect_surface_type(self, text_lower: str) -> str:
        for stype, patterns in SURFACE_TYPES.items():
            for p in patterns:
                if re.search(p, text_lower):
                    return stype
        return "unknown"

    def _generate_summary(self, fname: str, category: str, surface: str,
                          keywords: list[str], metadata: dict) -> str:
        cat_names = {
            "scalar_surface_integral": "scalar surface integral",
            "flux_integral": "flux (vector surface) integral",
            "divergence_theorem": "volume integral via Divergence Theorem",
            "stokes_theorem": "line/surface integral via Stokes' Theorem",
            "unknown": "surface integral problem",
        }
        cat_name = cat_names.get(category, category)

        surf_phrase = f" over a {surface}" if surface != "unknown" else ""
        kw_short = ", ".join(keywords[:5]) if keywords else "surface integral"

        pages = metadata.get("pages", "?")
        return (
            f"Problem {fname} is a {cat_name}{surf_phrase}. "
            f"The PDF has {pages} page(s) and involves: {kw_short}."
        )
