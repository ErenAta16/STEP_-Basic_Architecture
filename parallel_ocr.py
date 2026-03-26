"""
Parallel OCR stage shared by ``run.STEPSolver`` and ``pipeline.STEPPipeline``.

Nougat (L2) and the vision model (L3) are independent and I/O-heavy; when both are
enabled they run concurrently in a small thread pool. Skip rules (e.g. strong
PyMuPDF text so Nougat is not needed) match the previous inlined logic in ``run``
and ``pipeline``.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def run_parallel_nougat_vlm(
    *,
    pdf_path: Path,
    fname: str,
    img_dir: Path,
    nougat_layer: Any | None,
    vlm_layer: Any | None,
    use_nougat: bool,
    use_vlm: bool,
    vlm_available: bool,
    text_quality: dict,
    total_chars: int,
    verbose: bool,
    nougat_verbose: bool = True,
) -> dict[str, Any]:
    """Run Nougat and/or VLM in parallel and return structured results for callers.

    Args:
        pdf_path: PDF file being processed.
        fname: Basename stem (must match ``img_dir / fname / page_*.png`` for VLM).
        img_dir: Parent directory of per-PDF page image folders (from config).
        nougat_layer: ``Layer2_Nougat`` instance, or ``None`` when Nougat is disabled.
        vlm_layer: ``Layer3_VLM`` instance, or ``None`` when VLM is disabled.
        use_nougat: User flag to enable Nougat.
        use_vlm: User flag to enable VLM.
        vlm_available: ``True`` if the VLM layer has credentials and is usable.
        text_quality: Layer-0 ``analyze_text_quality`` dict (``score``, ``max_score``, …).
        total_chars: Total extracted text length (for Nougat skip heuristic).
        verbose: Emit progress lines via logging.
        nougat_verbose: Passed to ``nougat_layer.extract_from_pdf`` (per-page chatter).

    Returns:
        Dict with keys:

        - ``nougat_needed`` (bool): Whether Nougat was scheduled (``False`` if skipped
          by quality heuristic or disabled).
        - ``nougat_pkg``: ``None``, or ``(result_dict, quality_dict, elapsed_seconds)``
          from a successful Nougat run.
        - ``vlm_pkg``: ``None``, or ``(result_dict, quality_dict, elapsed_seconds)``
          from a successful VLM run.
        - ``t2_elapsed``, ``t3_elapsed`` (float): Last recorded wall times for L2/L3
          (``0.0`` if that branch did not run or failed before timing).

    Note:
        On failure inside a worker, the exception is logged and the corresponding
        ``*_pkg`` stays ``None``; callers mirror the old ``run``/``pipeline`` stub
        logging paths.
    """
    nougat_needed = use_nougat and nougat_layer is not None
    if nougat_needed and text_quality["score"] >= 6 and total_chars > 100:
        nougat_needed = False
        if verbose:
            _log.info(
                f"  [L2] Skipped Nougat (text quality {text_quality['score']}/"
                f"{text_quality['max_score']})"
            )

    nougat_pkg: tuple | None = None
    vlm_pkg: tuple | None = None
    t2_elapsed = 0.0
    t3_elapsed = 0.0

    futures: dict = {}

    def _consume_future(fut: Any) -> None:
        nonlocal nougat_pkg, vlm_pkg, t2_elapsed, t3_elapsed
        try:
            tag, res, q, elapsed = fut.result()
            if tag == "nougat":
                nougat_pkg = (res, q, elapsed)
                t2_elapsed = elapsed
            else:
                vlm_pkg = (res, q, elapsed)
                t3_elapsed = elapsed
        except Exception as e:
            _log.info(f"       [{futures[fut].upper()} FAIL] {str(e)[:60]}")

    def _run_nougat():
        t2 = time.time()
        if verbose:
            _log.info("  [L2] Nougat OCR...")
        res = nougat_layer.extract_from_pdf(pdf_path, verbose=nougat_verbose)
        q = nougat_layer.check_quality(res.get("latex", ""))
        elapsed = round(time.time() - t2, 2)
        if verbose:
            _log.info(
                f"       {res.get('char_count', 0)} chars, quality {q['score']}/"
                f"{q['max_score']} ({elapsed:.1f}s)"
            )
        return "nougat", res, q, elapsed

    def _run_vlm():
        t3 = time.time()
        vlm_label = f"{vlm_layer.provider or 'groq'}/{vlm_layer.model}"
        if verbose:
            _log.info(f"  [L3] VLM ({vlm_label})...")
        res = vlm_layer.extract_from_pdf_images(img_dir, fname, verbose=verbose)
        q = vlm_layer.check_quality(res.get("vlm_latex", ""))
        elapsed = round(time.time() - t3, 2)
        if verbose:
            _log.info(
                f"       {res.get('char_count', 0)} chars, quality {q['score']}/"
                f"{q['max_score']} ({elapsed:.1f}s)"
            )
        return "vlm", res, q, elapsed

    with ThreadPoolExecutor(max_workers=2) as pool:
        if nougat_needed:
            futures[pool.submit(_run_nougat)] = "nougat"
        if use_vlm and vlm_available and vlm_layer is not None:
            futures[pool.submit(_run_vlm)] = "vlm"

        for fut in as_completed(futures):
            _consume_future(fut)

    return {
        "nougat_needed": nougat_needed,
        "nougat_pkg": nougat_pkg,
        "vlm_pkg": vlm_pkg,
        "t2_elapsed": t2_elapsed,
        "t3_elapsed": t3_elapsed,
    }
