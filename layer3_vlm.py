"""
Layer 3 — vision path: page PNGs (from L0) → LaTeX.

Uses Gemini or Together Vision for image extraction. ``extract_from_pdf_images``
may run two independent VLM passes and pick the higher-quality output, unless
the first pass already achieves the maximum ``check_quality`` score (then pass 2
is skipped).
"""

import hashlib
import logging
import os
import re
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from config import (
    GEMINI_API_KEY,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
    TOGETHER_VLM_MODEL,
    TOGETHER_VLM_FALLBACK_MODEL,
    VLM_PROVIDER,
    VLM_OUT,
)

_log = logging.getLogger(__name__)

# Conservative default: two passes × N workers can burst the API (RPM limits).
# Set STEP_VLM_PAGE_WORKERS=1 if you see 429s or Gemini thread issues.
VLM_PAGE_WORKERS_DEFAULT = 2
GEMINI_VLM_MODEL = "gemini-2.5-flash"
VLM_NORMALIZE_LONG_EDGE = 2048

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
    """Vision extraction client (Gemini or Together)."""

    def __init__(self, force_provider: str | None = None):
        self.provider = None
        self.client = None
        self.model = None
        self._available = False
        self._gemini_client = None
        self._init(force_provider)

    def _init(self, force_provider: str | None = None):
        provider = (force_provider or VLM_PROVIDER or "gemini").strip().lower()
        if provider == "together":
            self._init_together()
            if not self._available:
                self._init_gemini()
            return
        # default / explicit gemini path
        self._init_gemini()
        if not self._available and provider != "gemini":
            self._init_together()

    def _init_together(self):
        if not TOGETHER_API_KEY:
            return
        try:
            self.client = OpenAI(api_key=TOGETHER_API_KEY, base_url=TOGETHER_BASE_URL)
            self.model = TOGETHER_VLM_MODEL
            self.provider = "together"
            self._available = True
        except Exception:
            pass

    def _init_gemini(self):
        if not GEMINI_API_KEY:
            return
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            if not self._available:
                self.model = GEMINI_VLM_MODEL
                self.provider = "gemini"
                self._available = True
        except Exception:
            pass

    def _ensure_gemini_client(self) -> bool:
        if self._gemini_client is not None:
            return True
        self._init_gemini()
        return self._gemini_client is not None

    @property
    def is_available(self) -> bool:
        return self._available

    @staticmethod
    def _prepare_image_bytes(image_path: str | Path) -> bytes:
        """Normalize scanned page images to reduce OCR variance across different PDF renders."""
        from PIL import Image as PILImage, ImageOps, ImageFilter

        img = PILImage.open(str(image_path)).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)

        w, h = img.size
        long_edge = max(w, h)
        if long_edge > 0 and long_edge != VLM_NORMALIZE_LONG_EDGE:
            scale = VLM_NORMALIZE_LONG_EDGE / float(long_edge)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), PILImage.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    @classmethod
    def _encode_image(cls, image_path: str | Path) -> str:
        return base64.b64encode(cls._prepare_image_bytes(image_path)).decode("utf-8")

    def extract_from_image(self, image_path: str | Path) -> str:
        """Single page image -> LaTeX problem statement (no solving)."""
        if not self._available:
            raise RuntimeError("VLM unavailable — set GEMINI_API_KEY or TOGETHER_API_KEY")
        if self.provider == "together":
            try:
                text = self._extract_together_with_model_fallback(image_path)
                if text and len(text.strip()) >= 40:
                    return text
                # Together Vision can return very short/empty text for unsupported or weak outputs.
                if self._ensure_gemini_client():
                    _log.info("  [L3] [WARN] Together VLM output too weak; falling back to Gemini VLM")
                    return self._extract_gemini(image_path)
                return ""
            except Exception as e:
                if self._ensure_gemini_client():
                    _log.info(f"  [L3] [WARN] Together VLM failed ({str(e)[:70]}); falling back to Gemini VLM")
                    return self._extract_gemini(image_path)
                raise
        return self._extract_gemini(image_path)

    def _extract_together(self, image_path: str | Path) -> str:
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

    def _extract_together_with_model_fallback(self, image_path: str | Path) -> str:
        try:
            return self._extract_together(image_path)
        except Exception as e:
            msg = str(e).lower()
            if (
                ("non-serverless model" in msg or "model_not_available" in msg)
                and TOGETHER_VLM_FALLBACK_MODEL
                and TOGETHER_VLM_FALLBACK_MODEL != self.model
            ):
                old = self.model
                self.model = TOGETHER_VLM_FALLBACK_MODEL
                _log.info(
                    f"  [L3] [WARN] Together model '{old}' unavailable; "
                    f"retrying with '{self.model}'"
                )
                return self._extract_together(image_path)
            raise

    def _extract_gemini(self, image_path: str | Path) -> str:
        from google.genai import types
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(self._prepare_image_bytes(image_path)))
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

    def _extract_pages(self, img_paths: list[Path], indices: list[int] | None = None,
                        verbose: bool = True) -> list[str]:
        """Run the VLM on each ``img_paths[i]`` for ``i`` in ``indices``; return per-page LaTeX.

        When ``indices`` is ``None``, every page runs (full pass). The returned
        list is page-aligned with ``img_paths``; entries for pages outside
        ``indices`` are left empty.
        """
        n = len(img_paths)
        parts = [""] * n
        if n == 0:
            return parts

        todo = list(range(n)) if indices is None else [i for i in indices if 0 <= i < n]
        if not todo:
            return parts

        try:
            w = int(os.environ.get("STEP_VLM_PAGE_WORKERS", str(VLM_PAGE_WORKERS_DEFAULT)))
        except ValueError:
            w = VLM_PAGE_WORKERS_DEFAULT
        workers = max(1, min(w, len(todo), 8))

        def one(idx: int) -> tuple[int, str]:
            try:
                return idx, self.extract_from_image(img_paths[idx])
            except Exception as e:
                if verbose:
                    _log.info(f"  [L3] [FAIL] {str(e)[:50]}")
                return idx, ""

        if workers == 1:
            for i in todo:
                _, latex = one(i)
                parts[i] = latex
            return parts

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, i) for i in todo]
            for fut in as_completed(futures):
                i, latex = fut.result()
                parts[i] = latex
        return parts

    def _extract_single_pass(self, img_paths: list[Path], verbose: bool = True) -> str:
        """Backward-compatible wrapper returning the concatenated text."""
        return "\n\n".join(self._extract_pages(img_paths, verbose=verbose))

    @staticmethod
    def _page_is_weak(page_text: str) -> bool:
        """True when a single-page pass1 output is short enough to warrant a retry."""
        if not page_text:
            return True
        t = page_text.strip()
        if len(t) < 25:
            return True
        # Almost no math content: no LaTeX commands, no digits → retry.
        if "\\" not in t and not re.search(r"\d", t):
            return True
        return False

    def _cache_paths(self, fname: str) -> tuple[Path, Path]:
        """Return ``(vlm.mmd, vlm.sha256)`` under the per-PDF VLM cache directory."""
        cache_dir = VLM_OUT / fname
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{fname}.vlm.mmd", cache_dir / f"{fname}.vlm.sha256"

    @staticmethod
    def _fingerprint_pages(img_paths: list[Path]) -> str:
        """Stable SHA-256 across all page PNGs (content + order).

        The VLM output depends on the exact pixel input, so DPI and normalization
        are implicitly included via the raw PNG bytes Layer 0 wrote.
        """
        h = hashlib.sha256()
        for p in img_paths:
            h.update(p.name.encode("utf-8"))
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(b"__missing__")
        return h.hexdigest()

    def extract_from_pdf_images(self, img_dir: Path, fname: str,
                                verbose: bool = True) -> dict:
        """Run a single full VLM pass, retry only weak pages, and cache the result.

        Optimizations vs. the previous "two full passes" approach:

        * Pass 1 runs on every page.
        * Pass 2 only re-runs pages whose individual output is empty/too short
          (``_page_is_weak``) — the old code re-ran *all* pages on a low overall
          quality score, which dominated total time on multi-page PDFs.
        * A disk cache under ``VLM_OUT/<fname>/<fname>.vlm.mmd`` short-circuits
          the entire VLM stage when the same pixel input is re-processed
          (identified by a SHA-256 over every page PNG).
        """
        page_dir = img_dir / fname
        if not page_dir.exists():
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        img_paths = _sorted_page_pngs(page_dir)
        if not img_paths:
            return {"file": fname, "vlm_latex": "", "char_count": 0, "pages": 0}

        fingerprint = self._fingerprint_pages(img_paths)
        mmd_path, sha_path = self._cache_paths(fname)
        if mmd_path.exists() and sha_path.exists():
            try:
                cached_sha = sha_path.read_text(encoding="utf-8").strip()
            except OSError:
                cached_sha = ""
            if cached_sha == fingerprint:
                try:
                    cached_text = mmd_path.read_text(encoding="utf-8")
                except OSError:
                    cached_text = ""
                if cached_text.strip():
                    if verbose:
                        _log.info(
                            f"  [L3] VLM cache hit ({len(cached_text)} chars, {len(img_paths)} page(s))"
                        )
                    return {
                        "file": fname,
                        "vlm_latex": cached_text,
                        "char_count": len(cached_text),
                        "raw_chars": [len(cached_text), 0],
                        "clean_chars": [len(cached_text), 0],
                        "pages": len(img_paths),
                        "cached": True,
                    }

        if verbose:
            _log.info("  [L3] VLM pass 1...")
        pages1 = self._extract_pages(img_paths, verbose=False)
        cleaned_pages = [self.clean_output(p) for p in pages1]

        weak_idx = [i for i, p in enumerate(cleaned_pages) if self._page_is_weak(p)]
        retry_n = 0
        if weak_idx and len(weak_idx) < len(img_paths):
            if verbose:
                _log.info(
                    f"  [L3] VLM pass 2 (retry {len(weak_idx)}/{len(img_paths)} weak page(s))..."
                )
            pages2 = self._extract_pages(img_paths, indices=weak_idx, verbose=False)
            for i in weak_idx:
                candidate = self.clean_output(pages2[i])
                if candidate and len(candidate) > len(cleaned_pages[i]):
                    cleaned_pages[i] = candidate
                    retry_n += 1
        elif weak_idx and len(weak_idx) == len(img_paths):
            # Every page looks weak — rerun all pages so the combined output still
            # has a chance at yielding usable LaTeX.
            if verbose:
                _log.info("  [L3] VLM pass 2 (retry all weak pages)...")
            pages2 = self._extract_pages(img_paths, verbose=False)
            for i in range(len(img_paths)):
                candidate = self.clean_output(pages2[i])
                if candidate and len(candidate) > len(cleaned_pages[i]):
                    cleaned_pages[i] = candidate
                    retry_n += 1

        full = "\n\n".join(p for p in cleaned_pages if p).strip()
        quality = self.check_quality(full)
        if verbose:
            _log.info(
                f"  [L3]    pages={len(img_paths)}, retries={retry_n}, "
                f"chars={len(full)} (quality {quality['score']}/{quality['max_score']})"
            )

        if full:
            try:
                mmd_path.write_text(full, encoding="utf-8")
                sha_path.write_text(fingerprint, encoding="utf-8")
            except OSError as e:
                if verbose:
                    _log.info(f"  [L3] [WARN] cache write failed: {str(e)[:60]}")

        return {
            "file": fname,
            "vlm_latex": full,
            "char_count": len(full),
            "raw_chars": [sum(len(p) for p in pages1), 0],
            "clean_chars": [len(full), 0],
            "pages": len(img_paths),
            "cached": False,
            "retries": retry_n,
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
