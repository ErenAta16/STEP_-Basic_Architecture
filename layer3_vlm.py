"""
Layer 3 — vision path: page PNGs → LaTeX via Groq (LLaMA 4 Scout multimodal).
"""

import re
import base64
from pathlib import Path

from openai import OpenAI

from config import GROQ_API_KEY, GROQ_BASE_URL

VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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


class Layer3_VLM:
    """Groq vision chat client; no-op if `GROQ_API_KEY` is empty."""

    def __init__(self):
        self.client = None
        self.model = VLM_MODEL
        self._available = False
        self._init()

    def _init(self):
        if not GROQ_API_KEY:
            return
        try:
            self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
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
        """Single page image → LaTeX problem statement (no solving)."""
        if not self._available:
            raise RuntimeError("VLM unavailable — set GROQ_API_KEY in .env")

        b64 = self._encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all mathematical content from this page as LaTeX:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _extract_single_pass(self, img_paths: list[Path], verbose: bool = True) -> str:
        """Run every page in order, concat with blank lines."""
        all_latex = []
        for img_path in img_paths:
            try:
                latex = self.extract_from_image(img_path)
                all_latex.append(latex)
            except Exception as e:
                if verbose:
                    print(f"[FAIL] {str(e)[:50]}")
                all_latex.append("")
        return "\n\n".join(all_latex)

    def extract_from_pdf_images(self, img_dir: Path, fname: str,
                                verbose: bool = True) -> dict:
        """Two independent passes; keep the one that scores higher on our quality rubric."""
        page_dir = img_dir / fname
        if not page_dir.exists():
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        img_paths = sorted(page_dir.glob("page_*.png"))
        if not img_paths:
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        if verbose:
            print(f"    VLM pass 1...", end=" ")
        raw1 = self._extract_single_pass(img_paths, verbose=False)
        clean1 = self.clean_output(raw1)
        if verbose:
            print(f"{len(clean1)} char", end="")

        if verbose:
            print(f" | pass 2...", end=" ")
        raw2 = self._extract_single_pass(img_paths, verbose=False)
        clean2 = self.clean_output(raw2)
        if verbose:
            print(f"{len(clean2)} char")

        if not clean1 and not clean2:
            full = ""
        elif not clean1:
            full = clean2
        elif not clean2:
            full = clean1
        else:
            q1 = self.check_quality(clean1)["score"]
            q2 = self.check_quality(clean2)["score"]
            if q1 > q2:
                full = clean1
            elif q2 > q1:
                full = clean2
            else:
                full = clean1 if len(clean1) >= len(clean2) else clean2

        if verbose:
            chosen = "pass1" if full == clean1 else "pass2"
            print(f"       Using {chosen} ({len(full)} chars)")

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
