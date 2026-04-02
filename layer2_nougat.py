"""
Layer 2 — Nougat: rasterized pages → LaTeX-ish markdown.

Registers a lightweight fake ``nougat.transforms`` so the real package does not
require ``albumentations``. Progress and readiness lines use ``logging``.
"""

import logging
import sys
import types
from pathlib import Path

import fitz
import torch
from PIL import Image
from torchvision import transforms as T

from config import NOUGAT_DPI
from layer0_ingestion import read_raster_sidecar, write_raster_sidecar

_log = logging.getLogger(__name__)


def _page_png_sort_key(p: Path) -> int:
    try:
        return int(p.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _sorted_page_pngs(img_dir: Path) -> list[Path]:
    return sorted(img_dir.glob("page_*.png"), key=_page_png_sort_key)


def _page_pngs_match_pdf(paths: list[Path], n_pages: int) -> bool:
    if n_pages < 1 or len(paths) != n_pages:
        return False
    for i, p in enumerate(paths):
        if _page_png_sort_key(p) != i + 1:
            return False
    return True


def _setup_albumentations_bypass():
    """Nougat import chain expects `nougat.transforms`; we stub it with torchvision."""
    fake_transforms = types.ModuleType("nougat.transforms")

    def _make_test_transform(config=None):
        input_size = (896, 672) if config is None else (config.input_size[1], config.input_size[2])
        return T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    fake_transforms.test_transform = _make_test_transform()
    fake_transforms.train_transform = None
    sys.modules["nougat.transforms"] = fake_transforms


def _patch_generate_validation():
    """Relax `GenerationMixin._validate_model_kwargs` for newer `transformers` releases."""
    from transformers.generation.utils import GenerationMixin
    original_validate = GenerationMixin._validate_model_kwargs

    def patched_validate(self, model_kwargs):
        model_kwargs.pop("encoder_outputs", None)
        try:
            return original_validate(self, model_kwargs)
        except ValueError:
            pass

    GenerationMixin._validate_model_kwargs = patched_validate


def _load_nougat_model():
    """Download checkpoint once and move model to CUDA if available."""
    import transformers
    if not hasattr(transformers.modeling_utils, 'PretrainedConfig'):
        from transformers import PretrainedConfig
        transformers.modeling_utils.PretrainedConfig = PretrainedConfig

    _patch_generate_validation()

    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint

    checkpoint = get_checkpoint("nougat")
    model = NougatModel.from_pretrained(checkpoint)

    if len(model.config.input_size) == 3:
        input_h = model.config.input_size[1]
        input_w = model.config.input_size[2]
    elif len(model.config.input_size) == 2:
        input_h = model.config.input_size[0]
        input_w = model.config.input_size[1]
    else:
        input_h, input_w = 896, 672

    transform = T.Compose([
        T.Resize((input_h, input_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, transform, device, (input_h, input_w)


class Layer2_Nougat:
    """Lazy-loads Nougat; writes `.mmd` next to rasterized pages."""

    def __init__(self, img_dir: str | Path, nougat_out: str | Path):
        self.img_dir = Path(img_dir)
        self.nougat_out = Path(nougat_out)
        self.model = None
        self.transform = None
        self.device = None
        self.disabled = False
        self.disabled_reason = ""

    def _disable(self, reason: str, verbose: bool = True) -> None:
        """Disable Nougat for the current process after a fatal runtime issue."""
        if self.disabled:
            return
        self.disabled = True
        self.disabled_reason = reason
        if verbose:
            _log.info(f"  [L2] Nougat disabled for this run: {reason}")

    @staticmethod
    def _empty_result(fname: str, pages: int = 0) -> dict:
        return {
            "file": fname,
            "latex": "",
            "char_count": 0,
            "pages": pages,
            "output_path": "",
            "cached": False,
            "disabled": True,
        }

    def initialize(self):
        """First call downloads weights — can take a while."""
        if self.model is not None:
            return

        _setup_albumentations_bypass()
        self.model, self.transform, self.device, input_size = _load_nougat_model()

        gpu_info = ""
        if torch.cuda.is_available():
            gpu_info = f" ({torch.cuda.get_device_name(0)})"
        _log.info(f"  [L2] Nougat ready on {self.device}{gpu_info}")
        _log.info(f"  [L2] Input size: {input_size[0]}x{input_size[1]}")

    def predict(self, img_path: str | Path) -> str:
        """Run inference on one PNG; returns markdown-ish LaTeX."""
        from nougat.postprocessing import markdown_compatible

        self.initialize()
        img = Image.open(str(img_path)).convert("RGB")
        pixel_values = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model.inference(image_tensors=pixel_values)

        raw = output["predictions"][0]
        processed = markdown_compatible(raw)

        if not processed.strip() and raw.strip():
            processed = raw.split("[repetition]")[0].strip()

        return processed

    def extract_from_pdf(self, pdf_path: str | Path, verbose: bool = True) -> dict:
        """Run Nougat on each page PNG, concatenate LaTeX.

        Reuses ``img_dir/<stem>/page_N.png`` from Layer 0 when page files match,
        ``.step_raster_meta`` records the same ``NOUGAT_DPI`` and the same PDF
        SHA-256 as this file. Otherwise rasterizes and refreshes the sidecar.
        Skips inference when a cached .mmd with matching PDF hash exists."""
        import hashlib

        pdf_path = Path(pdf_path)
        fname = pdf_path.stem
        if self.disabled:
            if verbose:
                _log.info(f"  [L2] Skipped Nougat (disabled): {self.disabled_reason}")
            return self._empty_result(fname)
        out_dir = self.nougat_out / fname
        out_dir.mkdir(parents=True, exist_ok=True)
        mmd_path = out_dir / f"{fname}.mmd"
        hash_path = out_dir / f"{fname}.sha256"

        pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

        if mmd_path.exists() and hash_path.exists():
            cached_hash = hash_path.read_text(encoding="utf-8").strip()
            if cached_hash == pdf_hash:
                full = mmd_path.read_text(encoding="utf-8")
                if verbose:
                    _log.info(f"  [L2] Nougat: cached ({len(full)} chars)")
                return {
                    "file": fname,
                    "latex": full,
                    "char_count": len(full),
                    "pages": 0,
                    "output_path": str(mmd_path),
                    "cached": True,
                }

        try:
            self.initialize()
        except Exception as e:
            self._disable(str(e), verbose=verbose)
            return self._empty_result(fname)
        img_dir = self.img_dir / fname
        img_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(str(pdf_path)) as doc:
            n_pages = doc.page_count
            candidate = _sorted_page_pngs(img_dir)
            side_dpi, side_hash = read_raster_sidecar(img_dir)
            meta_ok = (
                side_dpi is not None
                and side_hash is not None
                and side_dpi == NOUGAT_DPI
                and side_hash == pdf_hash
            )
            if meta_ok and _page_pngs_match_pdf(candidate, n_pages):
                img_paths = candidate
                if verbose:
                    _log.info(
                        f"  [L2] Reusing {len(img_paths)} page image(s) from Layer 0 "
                        f"(skip rasterize, DPI={NOUGAT_DPI}, hash OK)"
                    )
            else:
                if verbose and not meta_ok and candidate and _page_pngs_match_pdf(
                    candidate, n_pages
                ):
                    _log.info(
                        "  [L2] Raster sidecar missing or mismatch (DPI/PDF changed); re-rasterizing..."
                    )
                if verbose:
                    _log.info("  [L2] Rasterizing PDF pages...")
                img_paths = []
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=NOUGAT_DPI)
                    p = img_dir / f"page_{i + 1}.png"
                    pix.save(str(p))
                    img_paths.append(p)
                write_raster_sidecar(img_dir, NOUGAT_DPI, pdf_hash)

        all_latex = []
        for p in img_paths:
            pg = p.name
            try:
                latex = self.predict(p)
                all_latex.append(latex)
                if verbose:
                    _log.info(f"  [L2] Nougat: {pg}... [OK] {len(latex)} chars")
            except Exception as e:
                if verbose:
                    _log.info(f"  [L2] Nougat: {pg}... [FAIL] {e}")
                msg = str(e)
                if "meta tensors" in msg.lower():
                    self._disable(msg, verbose=verbose)
                    return self._empty_result(fname, pages=len(img_paths))
                all_latex.append("")

        full = "\n\n".join(all_latex)
        mmd_path.write_text(full, encoding="utf-8")
        hash_path.write_text(pdf_hash, encoding="utf-8")

        return {
            "file": fname,
            "latex": full,
            "char_count": len(full),
            "pages": len(img_paths),
            "output_path": str(mmd_path),
            "cached": False,
        }

    def check_quality(self, latex: str) -> dict:
        """Light checklist: did we get integrals, fractions, trig, enough length?"""
        checks = {
            "integral": "\\int" in latex,
            "frac": "\\frac" in latex,
            "trig": "\\sin" in latex or "\\cos" in latex,
            "content": len(latex.strip()) > 50,
        }
        return {
            "checks": checks,
            "score": sum(checks.values()),
            "max_score": len(checks),
        }
