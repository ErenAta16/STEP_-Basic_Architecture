"""
Layer 0: PyMuPDF (``fitz``) for metadata, per-page text, PNG tiles, and optional markdown.

Raster output feeds Nougat/VLM; text + markdown feed profiling and the LLM prompt.
"""

import hashlib
import logging
import time
import fitz  # PyMuPDF
from pathlib import Path

from config import NOUGAT_DPI

_log = logging.getLogger(__name__)

# Written next to page_*.png so Layer 2 can reuse rasters only when DPI and PDF match.
RASTER_SIDECAR_NAME = ".step_raster_meta"


def write_raster_sidecar(out_dir: Path, dpi: int, pdf_sha256: str) -> None:
    """Record DPI and PDF content hash for the PNGs in ``out_dir`` (newline-separated)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RASTER_SIDECAR_NAME).write_text(
        f"{dpi}\n{pdf_sha256}\n",
        encoding="utf-8",
    )


def read_raster_sidecar(out_dir: Path) -> tuple[int | None, str | None]:
    """Return ``(dpi, pdf_sha256)`` or ``(None, None)`` if missing or invalid."""
    p = out_dir / RASTER_SIDECAR_NAME
    if not p.exists():
        return None, None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return None, None
        return int(lines[0]), lines[1].strip()
    except (ValueError, OSError):
        return None, None


class Layer0_PDFIngestion:
    """Rasterize pages under ``img_dir/<pdf_stem>/`` and read text/metadata via explicit ``pdf_path``."""

    def __init__(self, img_dir: str | Path):
        self.img_dir = Path(img_dir)

    @staticmethod
    def _file_info(pdf_path: Path, doc: fitz.Document) -> dict:
        """Build the metadata dict (pages, author, on-disk size) for one open document."""
        meta = doc.metadata
        return {
            "file": pdf_path.name,
            "pages": doc.page_count,
            "author": meta.get("author", ""),
            "creator": meta.get("creator", ""),
            "producer": meta.get("producer", ""),
            "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
        }

    @staticmethod
    def _pages_text(doc: fitz.Document) -> list[dict]:
        """Extract plain ``get_text`` for each page (``page``, ``text``, ``char_count``)."""
        pages: list[dict] = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append({
                "page": i + 1,
                "text": text,
                "char_count": len(text),
            })
        return pages

    def _rasterize_pages(
        self, doc: fitz.Document, out_dir: Path, dpi: int
    ) -> list[dict]:
        """Write ``page_N.png`` under ``out_dir``; return image metadata rows."""
        out_dir.mkdir(parents=True, exist_ok=True)
        images: list[dict] = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            try:
                img_path = out_dir / f"page_{i + 1}.png"
                # On Windows, concurrent runs may briefly lock an existing PNG.
                # Retry a few times before failing hard.
                last_err = None
                for _ in range(4):
                    try:
                        pix.save(str(img_path))
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        time.sleep(0.15)
                if last_err is not None:
                    raise last_err
                images.append({
                    "page": i + 1,
                    "path": str(img_path),
                    "width": pix.width,
                    "height": pix.height,
                    "size_kb": round(img_path.stat().st_size / 1024, 1),
                })
            finally:
                pix = None
        return images

    def extract_metadata_text_and_images(
        self, pdf_path: str | Path, dpi: int | None = None
    ) -> tuple[dict, list[dict], list[dict]]:
        """Single ``fitz.open`` pass: metadata, text per page, and page PNGs under ``img_dir``.

        Uses ``alpha=False`` pixmaps (smaller files; fine for OCR/VLM).
        When ``dpi`` is omitted, uses ``NOUGAT_DPI`` so L0/L2 raster reuse stays aligned.
        """
        pdf_path = Path(pdf_path)
        if dpi is None:
            dpi = NOUGAT_DPI
        out_dir = self.img_dir / pdf_path.stem
        pdf_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        with fitz.open(str(pdf_path)) as doc:
            info = self._file_info(pdf_path, doc)
            pages = self._pages_text(doc)
            images = self._rasterize_pages(doc, out_dir, dpi)
        write_raster_sidecar(out_dir, dpi, pdf_sha256)
        return info, pages, images

    def extract_metadata_and_text(self, pdf_path: str | Path) -> tuple[dict, list[dict]]:
        """Metadata plus ``get_text`` per page without writing images."""
        pdf_path = Path(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            return self._file_info(pdf_path, doc), self._pages_text(doc)

    def extract_metadata(self, pdf_path: str | Path) -> dict:
        """Lightweight metadata PyMuPDF exposes (pages, producer, size)."""
        info, _ = self.extract_metadata_and_text(pdf_path)
        return info

    def extract_text(self, pdf_path: str | Path) -> list[dict]:
        """Plain `get_text` per page — good for native PDFs, weak on scans."""
        _, pages = self.extract_metadata_and_text(pdf_path)
        return pages

    def extract_markdown(
        self, pdf_path: str | Path, text_pages: list[dict] | None = None
    ) -> str:
        """Structure-aware markdown via PyMuPDF4LLM (headings/tables/layout).

        If ``pymupdf4llm`` is unavailable or raises, fall back to joining page
        texts. When ``text_pages`` is supplied (e.g. from the same run as L0
        ingest), that fallback needs no extra ``fitz.open``. If it is omitted,
        ``extract_text`` opens the PDF again to build the fallback string.
        """
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(pdf_path))
        except Exception:
            if text_pages is not None:
                return "\n\n".join(p["text"] for p in text_pages).strip()
            return "\n\n".join(p["text"] for p in self.extract_text(pdf_path)).strip()

    def extract_images(self, pdf_path: str | Path, dpi: int | None = None) -> list[dict]:
        """Rasterize each page to PNG under `img_dir/<stem>/` for Nougat/VLM.

        Prefer `extract_metadata_text_and_images` in hot paths to avoid a second
        PDF open when text is also needed."""
        pdf_path = Path(pdf_path)
        if dpi is None:
            dpi = NOUGAT_DPI
        out_dir = self.img_dir / pdf_path.stem
        pdf_sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        with fitz.open(str(pdf_path)) as doc:
            images = self._rasterize_pages(doc, out_dir, dpi)
        write_raster_sidecar(out_dir, dpi, pdf_sha256)
        return images

    def analyze_text_quality(self, pages: list[dict]) -> dict:
        """Cheap heuristics: did we lose integrals, superscripts, greek letters?

        Scans page-by-page to avoid building one giant string (lower peak RAM)."""
        greek_chars = "\u03d5\u03b8\u03b1\u03b2\u03b3\u03b4"
        super_chars = "\u00b2\u00b3\u2074\u2075"

        has_integral = has_partial = has_sqrt = has_pi = False
        has_greek = has_super = has_zz = False

        total_chars = 0
        for p in pages:
            t = p["text"]
            total_chars += len(t)
            if "\u222c" in t or "\u222b" in t:
                has_integral = True
            if "\u2202" in t:
                has_partial = True
            if "\u221a" in t:
                has_sqrt = True
            if "\u03c0" in t:
                has_pi = True
            if not has_greek:
                has_greek = any(c in t for c in greek_chars)
            if not has_super:
                has_super = any(c in t for c in super_chars)
            if "ZZ" in t:
                has_zz = True

        checks = {
            "integral_symbol": has_integral,
            "partial_deriv": has_partial,
            "sqrt_symbol": has_sqrt,
            "pi_symbol": has_pi,
            "greek_letters": has_greek,
            "superscripts": has_super,
            "no_garbled_ZZ": not has_zz,
        }

        issues = []
        if not checks["integral_symbol"]:
            issues.append("Integral symbol missing (often rendered as 'ZZ' or 'Z')")
        if not checks["superscripts"]:
            issues.append("Superscripts missing (x^2 flattened to x2)")
        if not checks["no_garbled_ZZ"]:
            issues.append("Double integral garbled as 'ZZ'")

        return {
            "checks": checks,
            "issues": issues,
            "score": sum(checks.values()),
            "max_score": len(checks),
            "total_chars": total_chars,
        }

    def process(self, pdf_path: str | Path, verbose: bool = True) -> dict:
        """One-shot helper: metadata, text quality, and page images (verbose CLI)."""
        pdf_path = Path(pdf_path)
        fname = pdf_path.name

        if verbose:
            _log.info(f"\n{'='*60}")
            _log.info(f"  {fname}")
            _log.info(f"{'='*60}")

        meta, pages, images = self.extract_metadata_text_and_images(pdf_path)
        if verbose:
            _log.info(f"\n  Metadata: {meta['pages']} pages, {meta['file_size_kb']} KB")
            _log.info(f"  Creator: {meta['creator']}")

        quality = self.analyze_text_quality(pages)

        if verbose:
            _log.info(f"\n  Baseline text extraction:")
            for pg in pages:
                preview = pg["text"][:120].replace("\n", " ").strip()
                _log.info(f"    Page {pg['page']}: {pg['char_count']} chars")
                _log.info(f"    Preview: '{preview}...'")

            _log.info(f"\n  Quality score: {quality['score']}/{quality['max_score']}")
            for key, val in quality["checks"].items():
                icon = "[OK]" if val else "[FAIL]"
                _log.info(f"    {icon} {key}")
            for issue in quality["issues"]:
                _log.info(f"    [!] {issue}")

        if verbose:
            _log.info(f"\n  Page images:")
            for img in images:
                _log.info(
                    f"    Page {img['page']}: {img['width']}x{img['height']}px, {img['size_kb']} KB"
                )

        return {
            "file": fname,
            "metadata": meta,
            "text_pages": pages,
            "quality": quality,
            "images": images,
        }
