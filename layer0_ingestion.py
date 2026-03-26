"""
Layer 0: PyMuPDF (``fitz``) for metadata, per-page text, PNG tiles, and optional markdown.

Raster output feeds Nougat/VLM; text + markdown feed profiling and the LLM prompt.
"""

import fitz  # PyMuPDF
from pathlib import Path


class Layer0_PDFIngestion:
    """Holds default PDF and image directories; methods accept explicit ``pdf_path``."""

    def __init__(self, pdf_dir: str | Path, img_dir: str | Path):
        self.pdf_dir = Path(pdf_dir)
        self.img_dir = Path(img_dir)

    def extract_metadata_text_and_images(
        self, pdf_path: str | Path, dpi: int = 300
    ) -> tuple[dict, list[dict], list[dict]]:
        """Single ``fitz.open`` pass: metadata, text per page, and page PNGs under ``img_dir``.

        Uses ``alpha=False`` pixmaps (smaller files; fine for OCR/VLM).
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        try:
            meta = doc.metadata
            info = {
                "file": pdf_path.name,
                "pages": doc.page_count,
                "author": meta.get("author", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
            }
            name = pdf_path.stem
            out_dir = self.img_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)

            pages: list[dict] = []
            images: list[dict] = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                try:
                    img_path = out_dir / f"page_{i + 1}.png"
                    pix.save(str(img_path))
                    images.append({
                        "page": i + 1,
                        "path": str(img_path),
                        "width": pix.width,
                        "height": pix.height,
                        "size_kb": round(img_path.stat().st_size / 1024, 1),
                    })
                finally:
                    # Drop reference so the pixmap buffer can be freed before the next page.
                    pix = None
            return info, pages, images
        finally:
            doc.close()

    def extract_metadata_and_text(self, pdf_path: str | Path) -> tuple[dict, list[dict]]:
        """Metadata plus ``get_text`` per page without writing images."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        try:
            meta = doc.metadata
            info = {
                "file": pdf_path.name,
                "pages": doc.page_count,
                "author": meta.get("author", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "file_size_kb": round(pdf_path.stat().st_size / 1024, 1),
            }
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                pages.append({
                    "page": i + 1,
                    "text": text,
                    "char_count": len(text),
                })
            return info, pages
        finally:
            doc.close()

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
        """Structure-aware markdown via PyMuPDF4LLM — preserves headings,
        tables, and basic layout better than plain get_text.
        Pass `text_pages` from a prior `extract_metadata_and_text` to avoid a
        third PDF open when pymupdf4llm fails."""
        try:
            import pymupdf4llm
            return pymupdf4llm.to_markdown(str(pdf_path))
        except Exception:
            if text_pages is not None:
                return "\n\n".join(p["text"] for p in text_pages).strip()
            return "\n\n".join(p["text"] for p in self.extract_text(pdf_path)).strip()

    def extract_images(self, pdf_path: str | Path, dpi: int = 300) -> list[dict]:
        """Rasterize each page to PNG under `img_dir/<stem>/` for Nougat/VLM.

        Prefer `extract_metadata_text_and_images` in hot paths to avoid a second
        PDF open when text is also needed."""
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        name = pdf_path.stem
        out_dir = self.img_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        images = []
        try:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                try:
                    img_path = out_dir / f"page_{i + 1}.png"
                    pix.save(str(img_path))
                    images.append({
                        "page": i + 1,
                        "path": str(img_path),
                        "width": pix.width,
                        "height": pix.height,
                        "size_kb": round(img_path.stat().st_size / 1024, 1),
                    })
                finally:
                    pix = None
        finally:
            doc.close()
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
            print(f"\n{'='*60}")
            print(f"  {fname}")
            print(f"{'='*60}")

        meta, pages, images = self.extract_metadata_text_and_images(pdf_path)
        if verbose:
            print(f"\n  Metadata: {meta['pages']} pages, {meta['file_size_kb']} KB")
            print(f"  Creator: {meta['creator']}")

        quality = self.analyze_text_quality(pages)

        if verbose:
            print(f"\n  Baseline text extraction:")
            for pg in pages:
                preview = pg["text"][:120].replace("\n", " ").strip()
                print(f"    Page {pg['page']}: {pg['char_count']} chars")
                print(f"    Preview: '{preview}...'")

            print(f"\n  Quality score: {quality['score']}/{quality['max_score']}")
            for key, val in quality["checks"].items():
                icon = "[OK]" if val else "[FAIL]"
                print(f"    {icon} {key}")
            for issue in quality["issues"]:
                print(f"    [!] {issue}")

        if verbose:
            print(f"\n  Page images:")
            for img in images:
                print(f"    Page {img['page']}: {img['width']}x{img['height']}px, {img['size_kb']} KB")

        return {
            "file": fname,
            "metadata": meta,
            "text_pages": pages,
            "quality": quality,
            "images": images,
        }
