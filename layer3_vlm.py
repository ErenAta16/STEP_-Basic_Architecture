"""
Layer 3 — vision path: page PNGs (from L0) → LaTeX.

Supports Groq (LLaMA 4 Scout) and Gemini. ``extract_from_pdf_images`` may run
two independent VLM passes and pick the higher-quality output, unless the first
pass already achieves the maximum ``check_quality`` score (then pass 2 is skipped).
"""

import logging
import os
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from config import GROQ_API_KEY, GROQ_BASE_URL, GEMINI_API_KEY

_log = logging.getLogger(__name__)

# Conservative default: two passes × N workers can burst the API (RPM limits).
# Set STEP_VLM_PAGE_WORKERS=1 if you see 429s or Gemini thread issues.
VLM_PAGE_WORKERS_DEFAULT = 2
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_VLM_MODEL = "gemini-2.5-flash"

VLM_SYSTEM_PROMPT = """You are a precise mathematical OCR tool. Your ONLY job is to read the mathematical problem from the image and output it in LaTeX.

ABSOLUTE RULES:
1. Output the COMPLETE problem statement in LaTeX. Include ALL given conditions:
   - The function/field definition (e.g. F = ...)
   - The surface/region description (e.g. S: x²+y²+z²=9)
   - Orientation or boundary conditions
   - Any constraints (e.g. z ≥ 0, first octant, outward)
2. NEVER solve, simplify, compute, derive, or show any work.
3. NEVER write "Step 1", "Step 2", or intermediate calculations.
4. NEVER include \\boxed{}, numerical answers, or final results.
5. Read EVERY symbol precisely — distinguish x^2 vs x^3, \\sin vs \\cos, + vs -.
6. Pay extreme attention to exponents: x^2 and x^3 look similar but are different.
7. Maximum output: 8 lines of LaTeX.

EXAMPLE INPUT: [image of a flux integral problem]
EXAMPLE OUTPUT:
\\text{Find the flux of } \\mathbf{F} = x^3\\,\\mathbf{i} + y^3\\,\\mathbf{j} + z^3\\,\\mathbf{k} \\text{ across } S: x^2+y^2+z^2=9 \\text{, outward orientation.}

If you include solutions, calculations, or numerical answers, you have FAILED."""


def _page_png_sort_key(p: Path) -> int:
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _sorted_page_pngs(page_dir: Path) -> list[Path]:
    return sorted(page_dir.glob("page_*.png"), key=_page_png_sort_key)


class Layer3_VLM:
    """Vision extraction client. Tries Gemini first (higher accuracy on math),
    falls back to Groq LLaMA Scout if Gemini key is missing."""

    def __init__(self, force_provider: str | None = None):
        self.provider = None
        self.client = None
        self.model = None
        self._available = False
        self._gemini_client = None
        self._init(force_provider)

    def _init(self, force_provider: str | None = None):
        if force_provider == "groq" or (force_provider is None and not GEMINI_API_KEY):
            self._init_groq()
        elif force_provider == "gemini" or (force_provider is None and GEMINI_API_KEY):
            self._init_gemini()
            if not self._available:
                self._init_groq()

    def _init_groq(self):
        if not GROQ_API_KEY:
            return
        try:
            self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            self.model = VLM_MODEL
            self.provider = "groq"
            self._available = True
        except Exception:
            pass

    def _init_gemini(self):
        if not GEMINI_API_KEY:
            return
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            self.model = GEMINI_VLM_MODEL
            self.provider = "gemini"
            self._available = True
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        with open(str(image_path), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_from_image(self, image_path: str | Path) -> str:
        """Single page image -> LaTeX problem statement (no solving)."""
        if not self._available:
            raise RuntimeError("VLM unavailable — set GEMINI_API_KEY or GROQ_API_KEY")

        if self.provider == "gemini":
            return self._extract_gemini(image_path)
        return self._extract_groq(image_path)

    def _extract_groq(self, image_path: str | Path) -> str:
        b64 = self._encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all mathematical content from this page as LaTeX:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _extract_gemini(self, image_path: str | Path) -> str:
        from google.genai import types
        from PIL import Image as PILImage

        img = PILImage.open(str(image_path))
        response = self._gemini_client.models.generate_content(
            model=self.model,
            contents=[
                VLM_SYSTEM_PROMPT + "\n\nExtract all mathematical content from this page as LaTeX:",
                img,
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=4096,
                temperature=0.0,
            ),
        )
        return response.text or ""

    def _extract_single_pass(self, img_paths: list[Path], verbose: bool = True) -> str:
        """Run every page in order, concat with blank lines. Uses a small thread pool for I/O-bound APIs."""
        n = len(img_paths)
        if n == 0:
            return ""

        try:
            w = int(os.environ.get("STEP_VLM_PAGE_WORKERS", str(VLM_PAGE_WORKERS_DEFAULT)))
        except ValueError:
            w = VLM_PAGE_WORKERS_DEFAULT
        workers = max(1, min(w, n, 8))

        def one(idx: int) -> tuple[int, str]:
            try:
                return idx, self.extract_from_image(img_paths[idx])
            except Exception as e:
                if verbose:
                    _log.info(f"  [L3] [FAIL] {str(e)[:50]}")
                return idx, ""

        if workers == 1:
            parts = [""] * n
            for i in range(n):
                _, latex = one(i)
                parts[i] = latex
            return "\n\n".join(parts)

        parts = [""] * n
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, i) for i in range(n)]
            for fut in as_completed(futures):
                i, latex = fut.result()
                parts[i] = latex
        return "\n\n".join(parts)

    def extract_from_pdf_images(self, img_dir: Path, fname: str,
                                verbose: bool = True) -> dict:
        """Run one or two VLM passes over page PNGs and pick the best LaTeX by quality score.

        Pass 2 is skipped when pass 1 already reaches ``max_score`` on
        ``check_quality`` (saves latency and API cost). Otherwise both passes
        run; the higher-scoring (or longer, on tie) cleaned string wins.
        """
        page_dir = img_dir / fname
        if not page_dir.exists():
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        img_paths = _sorted_page_pngs(page_dir)
        if not img_paths:
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        if verbose:
            _log.info("  [L3] VLM pass 1...")
        raw1 = self._extract_single_pass(img_paths, verbose=False)
        clean1 = self.clean_output(raw1)
        q1 = self.check_quality(clean1)
        if verbose:
            _log.info(
                f"  [L3]    pass 1: {len(clean1)} char (quality {q1['score']}/{q1['max_score']})"
            )

        if q1["score"] == q1["max_score"] and clean1:
            if verbose:
                _log.info(f"  [L3]    Using pass1 (max quality, {len(clean1)} chars)")
            return {
                "file": fname,
                "vlm_latex": clean1,
                "char_count": len(clean1),
                "raw_chars": [len(raw1), 0],
                "clean_chars": [len(clean1), 0],
                "pages": len(img_paths),
            }

        if verbose:
            _log.info("  [L3] VLM pass 2...")
        raw2 = self._extract_single_pass(img_paths, verbose=False)
        clean2 = self.clean_output(raw2)
        if verbose:
            q2p = self.check_quality(clean2)
            _log.info(
                f"  [L3]    pass 2: {len(clean2)} char (quality {q2p['score']}/{q2p['max_score']})"
            )

        if not clean1 and not clean2:
            full = ""
        elif not clean1:
            full = clean2
        elif not clean2:
            full = clean1
        else:
            s1 = q1["score"]
            s2 = self.check_quality(clean2)["score"]
            if s1 > s2:
                full = clean1
            elif s2 > s1:
                full = clean2
            else:
                full = clean1 if len(clean1) >= len(clean2) else clean2

        if verbose:
            chosen = "pass1" if full == clean1 else "pass2"
            _log.info(f"  [L3]    Using {chosen} ({len(full)} chars)")

        return {
            "file": fname,
            "vlm_latex": full,
            "char_count": len(full),
            "raw_chars": [len(raw1), len(raw2)],
            "clean_chars": [len(clean1), len(clean2)],
            "pages": len(img_paths),
        }

    @staticmethod
    def clean_output(text: str) -> str:
        """Strip boxed answers / “final answer” tails so L5 sees the statement only."""
        if not text:
            return ""

        text = re.sub(r'\\boxed\{[^}]*\}', '', text)
        text = re.sub(r'The final answer is:?.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'FINAL_ANSWER:.*', '', text, flags=re.IGNORECASE)

        lines = text.split("\n")
        cleaned = []

        solution_phrases = [
            "step ", "## step", "recall that", "apply the",
            "we are left", "we get", "we have", "we can",
            "we need to", "we know", "we use", "we obtain",
            "let me", "i can", "i will",
            "note:", "note that", "answer:", "solution:",
            "therefore,", "thus,", "hence,", "finally,",
            "substituting", "plugging", "computing",
            "after simplif", "after evaluat",
            "this gives us", "this yields", "this means that",
            "this equals", "this simplifies",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()

            if re.match(r'^#{1,4}\s*\d+', stripped):
                continue

            is_solution_step = any(
                lower.startswith(p) or lower.lstrip('#').strip().lower().startswith(p)
                for p in solution_phrases
            )
            if is_solution_step:
                continue

            has_equals_chain = bool(re.search(r'=.*=.*=', stripped))
            if has_equals_chain and not any(kw in lower for kw in ["find", "compute", "evaluate", "flux", "integral"]):
                continue

            cleaned.append(stripped)

        result = "\n".join(cleaned).strip()
        result = re.sub(r'\n{3,}', '\n\n', result)

        if len(result) > 2000:
            lines = result.split("\n")
            result = "\n".join(lines[:12]).strip()

        return result

    def check_quality(self, vlm_latex: str) -> dict:
        """Same idea as Nougat scoring: integrals, fractions, length, backslashes."""
        checks = {
            "integral": "\\int" in vlm_latex or "\\iint" in vlm_latex,
            "frac": "\\frac" in vlm_latex,
            "math_content": len(vlm_latex.strip()) > 50,
            "latex_commands": "\\" in vlm_latex,
        }
        return {
            "checks": checks,
            "score": sum(checks.values()),
            "max_score": len(checks),
        }
