"""
STEP Pipeline — older CLI (`pipeline.STEPPipeline`).

Usage:
    python main.py                     # full pipeline, first 5 PDFs
    python main.py --count 10
    python main.py --pdf-dir ./my_pdfs
    python main.py --layer0            # Layer 0 only
    python main.py --layer2            # Nougat only
    python main.py --layer6            # SymPy tests only
    python main.py --full
    python main.py --check-gpu
    python main.py --list-pdfs
"""

import argparse
import sys
from pathlib import Path

import torch

from config import PDF_DIR


def check_gpu():
    """Print whether CUDA is available (Nougat cares a lot)."""
    print("=" * 50)
    print("  GPU CHECK")
    print("=" * 50)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [OK] GPU: {gpu_name}")
        print(f"       VRAM: {gpu_mem:.1f} GB")
    else:
        print("  [!] No CUDA GPU")
        print("      Layer 0 and 6 run on CPU; Nougat is much slower without a GPU.")


def list_pdfs(pdf_dir: Path):
    """List first few PDFs in a folder."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"\n  {pdf_dir}: {len(pdf_files)} PDF(s)")
    for f in pdf_files[:10]:
        size = f.stat().st_size / 1024
        print(f"    {f.name} ({size:.1f} KB)")
    if len(pdf_files) > 10:
        print(f"    ... and {len(pdf_files) - 10} more")
    if not pdf_files:
        print(f"  [!] No PDFs — drop files into {pdf_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="STEP Pipeline — surface integral PDF solver",
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
        "--layer0",
        action="store_true",
        help="Run Layer 0 (ingest) test only",
    )
    parser.add_argument(
        "--layer2",
        action="store_true",
        help="Run Layer 2 (Nougat) test only",
    )
    parser.add_argument(
        "--layer6",
        action="store_true",
        help="Run Layer 6 (SymPy) standalone tests only",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full pipeline: Nougat → LLM → SymPy",
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

    args = parser.parse_args()
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else PDF_DIR

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║   STEP Pipeline — local run          ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    if args.check_gpu:
        check_gpu()
        return

    if args.list_pdfs:
        list_pdfs(pdf_dir)
        return

    from pipeline import STEPPipeline
    pipeline = STEPPipeline(pdf_dir=pdf_dir, provider=args.provider,
                            ensemble=args.ensemble)

    no_specific_layer = not (args.layer0 or args.layer2 or args.layer6 or args.full)

    if args.layer0:
        pipeline.run_layer0_test(count=args.count)

    if args.layer2:
        pipeline.run_layer2_test(count=args.count)

    if args.layer6:
        from layer6_verifier import run_standalone_tests
        run_standalone_tests()

    if args.full or no_specific_layer:
        check_gpu()
        list_pdfs(pdf_dir)
        print()
        pipeline.run_full_pipeline(count=args.count)


if __name__ == "__main__":
    main()
