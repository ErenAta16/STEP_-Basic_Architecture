"""
Alternate entry point: runs ``pipeline.STEPPipeline`` on a folder with JSON logging.

``run.py`` is preferred for one-off PDFs; this script is for batch + ``pipeline_log_*.json``.
Calls ``ensure_dirs`` and ``configure_logging`` at startup so paths and log output match ``run``.

Usage:
    python main.py
    python main.py -n 10
    python main.py --pdf-dir ./my_pdfs
    python main.py --check-gpu
    python main.py --list-pdfs
    python main.py --no-nougat
    python main.py --no-vlm
"""

import argparse
import logging
from pathlib import Path

import torch

from config import PDF_DIR, ensure_dirs
from step_logging import configure_logging

_log = logging.getLogger(__name__)


def check_gpu():
    """Show CUDA device name/VRAM if present (Nougat is slow without a GPU)."""
    _log.info("=" * 50)
    _log.info("  GPU CHECK")
    _log.info("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        _log.info(f"  [OK] GPU: {gpu_name}")
        _log.info(f"       VRAM: {gpu_mem:.1f} GB")
    else:
        _log.info("  [!] No CUDA GPU")
        _log.info("      Layer 0 and 6 run on CPU; Nougat is much slower without a GPU.")


def list_pdfs(pdf_dir: Path):
    """Print a short listing (up to 10 files) for the given directory."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    _log.info(f"\n  {pdf_dir}: {len(pdf_files)} PDF(s)")
    for f in pdf_files[:10]:
        size = f.stat().st_size / 1024
        _log.info(f"    {f.name} ({size:.1f} KB)")
    if len(pdf_files) > 10:
        _log.info(f"    ... and {len(pdf_files) - 10} more")
    if not pdf_files:
        _log.info(f"  [!] No PDFs - drop files into {pdf_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="STEP Pipeline - surface integral PDF solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help=f"Folder with PDFs (default: {PDF_DIR})",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="How many PDFs to process (default: 5)",
    )
    parser.add_argument(
        "--check-gpu",
        action="store_true",
        help="Print GPU availability",
    )
    parser.add_argument(
        "--list-pdfs",
        action="store_true",
        help="List PDFs in the folder",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["groq", "gemini", "claude", "openai"],
        help="Force one LLM backend (default: auto)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Run every configured LLM and pick best (experimental)",
    )
    parser.add_argument(
        "--with-nougat",
        action="store_true",
        help="Enable Nougat (default is off for this environment)",
    )
    parser.add_argument(
        "--no-nougat",
        action="store_true",
        help="Deprecated alias; Nougat is already off by default",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Full pipeline: skip vision model (Nougat-only if enabled, else raw fallback)",
    )

    args = parser.parse_args()
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else PDF_DIR

    ensure_dirs()
    configure_logging()

    _log.info("")
    _log.info("  ======================================")
    _log.info("  STEP Pipeline - local run")
    _log.info("  ======================================")
    _log.info("")

    if args.check_gpu:
        check_gpu()
        return

    if args.list_pdfs:
        list_pdfs(pdf_dir)
        return

    from pipeline import STEPPipeline
    pipeline = STEPPipeline(
        pdf_dir=pdf_dir,
        provider=args.provider,
        ensemble=args.ensemble,
        use_nougat=args.with_nougat and not args.no_nougat,
        use_vlm=not args.no_vlm,
    )

    check_gpu()
    list_pdfs(pdf_dir)
    _log.info("")
    pipeline.run_full_pipeline(count=args.count)


if __name__ == "__main__":
    main()
