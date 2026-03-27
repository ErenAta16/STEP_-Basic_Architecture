"""
CLI entry for the STEP PDF -> LLM pipeline (``STEPSolver``).

Uses ``logging`` (via ``step_logging.configure_logging``) for stdout-friendly
lines and ``config.ensure_dirs`` before touching pipeline paths.

Usage:
    python run.py problem.pdf
    python run.py problem.pdf --no-nougat
    python run.py Surface_Integration/
    python run.py Surface_Integration/ -n 5
    python run.py --check
"""

import argparse
import logging
import sys
import time
import platform
from collections import OrderedDict
from pathlib import Path

import torch

from config import (PDF_DIR, IMG_DIR, NOUGAT_OUT, RESULTS_DIR, NOUGAT_DPI,
                    get_system_prompt, ensure_dirs)
from parallel_ocr import run_parallel_nougat_vlm
from step_logging import configure_logging
from layer0_ingestion import Layer0_PDFIngestion
from layer1_profiler import Layer1_Profiler
from layer2_nougat import Layer2_Nougat
from layer3_vlm import Layer3_VLM
from layer4_synthesis import Layer4_Synthesis
from layer5_llm_solver import Layer5_LLMSolver
from layer6_verifier import Layer6_SymPyVerifier
from pipeline_logger import PipelineLogger

_log = logging.getLogger(__name__)


class STEPSolver:
    """Orchestrates layers 0–6: ingest, profile, optional Nougat/VLM (via ``parallel_ocr``), synthesis, LLM, extract."""

    def __init__(self, use_nougat: bool = True, use_vlm: bool = True):
        self.l0 = Layer0_PDFIngestion(IMG_DIR)
        self.l1 = Layer1_Profiler()
        self.l2 = Layer2_Nougat(IMG_DIR, NOUGAT_OUT) if use_nougat else None
        self.l3 = Layer3_VLM() if use_vlm else None
        self.l4 = Layer4_Synthesis()
        self.l5_primary = Layer5_LLMSolver(force_provider="gemini")
        self.l5_fallback = Layer5_LLMSolver(force_provider="groq")
        self.l5 = self.l5_primary if self.l5_primary.is_available else self.l5_fallback
        self.l6 = Layer6_SymPyVerifier()
        self.use_nougat = use_nougat
        self.use_vlm = use_vlm
        self._result_cache: OrderedDict[str, dict] = OrderedDict()
        self._result_cache_max = 32

    @staticmethod
    def _pdf_hash(pdf_path: Path) -> str:
        import hashlib
        return hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    def solve(self, pdf_path: str | Path, verbose: bool = True) -> dict:
        """Run all enabled stages on one PDF.

        On a cache miss, ensures directories exist and attaches logging to stdout.
        On success the dict includes timings, solution text, and ``final_answer``
        (display string). On failure look for an ``error`` key; partial timings
        may still be present.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        h = self._pdf_hash(pdf_path)
        path_key = str(pdf_path.resolve())
        # Per resolved path + content hash + OCR flags (same bytes at two paths -> separate entries).
        cache_key = f"{path_key}|{h}|nougat={int(self.use_nougat)}|vlm={int(self.use_vlm)}"
        if cache_key in self._result_cache:
            if verbose:
                _log.info(f"  [CACHE] Returning previous result for {pdf_path.name}")
            self._result_cache.move_to_end(cache_key)
            return self._result_cache[cache_key]

        ensure_dirs()
        configure_logging()

        fname = pdf_path.stem
        result = {"file": fname, "pdf_path": str(pdf_path)}
        t_start = time.time()
        timings = {}

        # --- L0: PDF Ingestion ---
        t0 = time.time()
        metadata, raw_pages, _img_meta = self.l0.extract_metadata_text_and_images(
            pdf_path, dpi=NOUGAT_DPI
        )
        raw_text = "\n".join(p["text"] for p in raw_pages).strip()
        md_text = self.l0.extract_markdown(pdf_path, text_pages=raw_pages)
        text_quality = self.l0.analyze_text_quality(raw_pages)
        result["pages"] = metadata.get("pages", 0)
        result["chars"] = sum(len(p["text"]) for p in raw_pages)
        result["text_quality_score"] = text_quality["score"]
        timings["l0_ingest"] = round(time.time() - t0, 3)
        if verbose:
            _log.info(
                f"  [L0] {metadata.get('pages')} pages, {result['chars']} chars, "
                f"quality {text_quality['score']}/{text_quality['max_score']} ({timings['l0_ingest']:.2f}s)"
            )

        # --- L1: quick tags from raw text (cheap classifier) ---
        t1 = time.time()
        profile = self.l1.profile(fname, metadata, raw_text)
        timings["l1_profile"] = round(time.time() - t1, 4)
        if verbose:
            sec = profile.get("secondary_categories") or []
            sec_s = f" | also: {', '.join(sec)}" if sec else ""
            _log.info(
                f"  [L1] {profile['category']} / {profile['surface_type']}{sec_s} "
                f"({timings['l1_profile']:.3f}s)"
            )

        nougat_latex = ""
        nougat_score = 0
        vlm_latex = ""
        vlm_score = 0
        timings["l2_nougat"] = 0.0
        timings["l3_vlm"] = 0.0

        # Shared with ``pipeline.STEPPipeline`` (skip rules + ThreadPoolExecutor).
        ocr = run_parallel_nougat_vlm(
            pdf_path=pdf_path,
            fname=fname,
            img_dir=IMG_DIR,
            nougat_layer=self.l2,
            vlm_layer=self.l3,
            use_nougat=self.use_nougat,
            use_vlm=self.use_vlm,
            vlm_available=self.l3 is not None and self.l3.is_available,
            text_quality=text_quality,
            total_chars=result["chars"],
            verbose=verbose,
            nougat_verbose=verbose,
        )
        if ocr["nougat_pkg"]:
            res_n, q_n, _ = ocr["nougat_pkg"]
            nougat_latex = res_n.get("latex", "")
            nougat_score = q_n["score"]
        timings["l2_nougat"] = ocr["t2_elapsed"]
        if ocr["vlm_pkg"]:
            res_v, q_v, _ = ocr["vlm_pkg"]
            vlm_latex = res_v.get("vlm_latex", "")
            vlm_score = q_v["score"]
        timings["l3_vlm"] = ocr["t3_elapsed"]

        # --- L1b: Re-profile with OCR/VLM text if raw was empty ---
        if result["chars"] == 0 and (vlm_latex or nougat_latex):
            ocr_text = vlm_latex or nougat_latex
            profile = self.l1.profile(fname, metadata, raw_text, latex_text=ocr_text)
            if verbose:
                _log.info(f"  [L1b] Re-profiled: {profile['category']} / {profile['surface_type']}")

        result["category"] = profile["category"]
        result["surface_type"] = profile["surface_type"]
        result["keywords"] = profile["keywords"]
        result["summary"] = profile["summary"]
        result["secondary_categories"] = profile.get("secondary_categories", [])

        # --- L4: Synthesis ---
        t4 = time.time()
        synthesis = self.l4.synthesize(raw_text, nougat_latex, nougat_score,
                                        vlm_latex, vlm_score, profile,
                                        md_text=md_text)
        prompt = synthesis["prompt"]
        domain = synthesis.get("domain", "general_math")
        result["source"] = synthesis["source"]
        result["domain"] = domain
        result["nougat_score"] = nougat_score
        result["vlm_score"] = vlm_score
        result["prompt_chars"] = synthesis.get("prompt_chars", len(prompt))
        timings["l4_synthesis"] = round(time.time() - t4, 4)
        if verbose:
            domain_label = "surface_integral" if domain == "surface_integral" else "general_math"
            _log.info(
                f"  [L4] source={synthesis['source']}, domain={domain_label}, "
                f"prompt_chars={synthesis['prompt_chars']}"
            )

        # --- L5: LLM Solve with Consensus (2-3 attempts) ---
        system_prompt = get_system_prompt(
            domain,
            secondary_categories=profile.get("secondary_categories"),
            primary_category=profile.get("category"),
        )
        result["l5_system"] = {
            "domain": domain,
            "secondary_categories": list(profile.get("secondary_categories") or []),
        }
        if verbose:
            dom_tag = "surface" if domain == "surface_integral" else "general"
            sec = profile.get("secondary_categories") or []
            sec_part = f", signals={sec}" if sec else ""
            _log.info(f"  [L5] system={dom_tag}{sec_part}")
        t5 = time.time()
        answers = self._solve_with_consensus(prompt, system_prompt, verbose=verbose)
        timings["l5_llm"] = round(time.time() - t5, 2)
        result["attempts"] = len(answers)

        if not answers or not any(a.get("solution") for a in answers):
            result["error"] = "All LLM attempts failed"
            result["timings"] = timings
            result["elapsed_s"] = round(time.time() - t_start, 1)
            return result

        best = next((a for a in answers if a.get("solution")), answers[0])
        result["solution"] = best.get("solution", "")
        result["solution_chars"] = len(best.get("solution", ""))
        result["consensus"] = best.get("consensus", False)
        result["model_used"] = best.get("model", "")

        # --- L6: extract FINAL_ANSWER / boxed for display ---
        t6 = time.time()
        fa = self.l6._extract_final_answer(best["solution"])
        result["final_answer"] = fa if fa else "(could not extract)"
        if verbose:
            tag = "[OK]" if fa else "[?]"
            _log.info(f"  [L6] {tag} extracted final line for display")

        # --- optional SUMMARY: block for the UI ---
        result["llm_summary"] = self._extract_llm_summary(best["solution"])

        timings["l6_extract"] = round(time.time() - t6, 4)
        result["timings"] = timings
        result["elapsed_s"] = round(time.time() - t_start, 1)

        self._result_cache[cache_key] = result
        self._result_cache.move_to_end(cache_key)
        while len(self._result_cache) > self._result_cache_max:
            self._result_cache.popitem(last=False)
        return result

    def _solve_with_consensus(self, prompt: str, system_prompt: str,
                               max_attempts: int = 3, verbose: bool = True) -> list[dict]:
        """Up to ``max_attempts`` solves, rotating Gemini (primary) and Groq (fallback).

        Parses each reply to a numeric value when possible and stops early if two
        attempts agree. Handles 429/503-style errors by backing off and switching provider.
        """
        from latex_parser import parse_latex_to_value

        INTER_ATTEMPT_DELAY = 3
        RETRY_503_DELAY = 8

        solvers = []
        if self.l5_primary.is_available:
            solvers.append(("primary", self.l5_primary))
        if self.l5_fallback.is_available:
            solvers.append(("fallback", self.l5_fallback))

        attempts = []
        answer_counts = {}
        rate_limited = set()
        consecutive_503 = 0

        for i in range(max_attempts):
            if i > 0 and consecutive_503 == 0:
                time.sleep(INTER_ATTEMPT_DELAY)

            solver = None
            solver_label = ""
            for tag, s in solvers:
                if tag not in rate_limited:
                    solver = s
                    solver_label = f"{s.provider}/{s.model_name}"
                    break
            if solver is None:
                if verbose:
                    _log.info(f"  [L5] All models busy; sleeping {RETRY_503_DELAY}s...")
                time.sleep(RETRY_503_DELAY)
                rate_limited.clear()
                consecutive_503 = 0
                solver = solvers[0][1]
                solver_label = f"{solver.provider}/{solver.model_name}"

            t5 = time.time()
            try:
                solution = solver.solve(prompt, system_prompt=system_prompt)
                consecutive_503 = 0
                fa = self.l6._extract_final_answer(solution)
                elapsed = time.time() - t5

                fa_num = parse_latex_to_value(fa) if fa else None
                fa_key = f"{fa_num:.6f}" if fa_num is not None else (fa if fa else None)

                attempts.append({
                    "solution": solution,
                    "final_answer": fa,
                    "numeric": fa_num,
                    "key": fa_key,
                    "elapsed": elapsed,
                    "model": solver_label,
                })

                if fa_key is not None:
                    answer_counts[fa_key] = answer_counts.get(fa_key, 0) + 1

                if verbose:
                    _log.info(
                        f"  [L5] {solver_label} [{i+1}/{max_attempts}]... "
                        f"-> {fa or '?'} ({elapsed:.1f}s)"
                    )

                if i > 0 and fa_key is not None and answer_counts.get(fa_key, 0) >= 2:
                    if verbose:
                        _log.info(
                            f"  [L5] [OK] Consensus: {fa} ({answer_counts[fa_key]}/{i+1} agree)"
                        )
                    for a in attempts:
                        if a["key"] == fa_key:
                            a["consensus"] = True
                    return [a for a in attempts if a["key"] == fa_key][:1]

            except Exception as e:
                err = str(e)
                if verbose:
                    _log.info(f"[ERR] {err[:80]}")

                is_transient = any(k in err for k in ["503", "UNAVAILABLE", "overloaded", "high demand"])
                is_rate = any(k in err for k in ["429", "RESOURCE_EXHAUSTED", "rate", "quota"])
                is_fatal = any(k in err for k in ["404", "NOT_FOUND"])

                if is_transient and consecutive_503 < 2:
                    consecutive_503 += 1
                    delay = RETRY_503_DELAY * consecutive_503
                    if verbose:
                        _log.info(f"  [L5] [RETRY] HTTP 503: same model again in {delay}s...")
                    time.sleep(delay)
                elif is_rate or is_fatal or (is_transient and consecutive_503 >= 2):
                    consecutive_503 = 0
                    for tag, s in solvers:
                        if s is solver:
                            rate_limited.add(tag)
                            if verbose:
                                _log.info(f"  [L5] [WARN] {solver_label}: switching to fallback")
                            break

                attempts.append({"solution": "", "final_answer": "", "error": err})

        if not attempts:
            return []

        valid_counts = {k: v for k, v in answer_counts.items() if k is not None}
        if valid_counts:
            best_key = max(valid_counts, key=valid_counts.get)
            best_count = valid_counts[best_key]
            if verbose and best_count < 2:
                _log.info(
                    f"  [L5] [WARN] Weak consensus; best guess: {best_key} "
                    f"({best_count}/{len(attempts)})"
                )
            result = [a for a in attempts if a.get("key") == best_key]
            if result:
                return result

        valid = [a for a in attempts if a.get("final_answer")]
        if valid:
            return valid[:1]
        valid = [a for a in attempts if a.get("solution")]
        return valid[:1] if valid else attempts[:1]

    @staticmethod
    def _extract_llm_summary(solution: str) -> dict:
        """Parse the structured SUMMARY: block from the system prompt template (if present)."""
        import re
        summary = {}
        idx = solution.find("SUMMARY:")
        if idx == -1:
            return summary
        block = solution[idx:]
        fields = [
            ("problem_type", r"Problem Type:\s*(.+)"),
            ("method_used", r"Method Used:\s*(.+)"),
            ("surface", r"Surface:\s*(.+)"),
            ("key_steps", r"Key Steps:\s*(.+)"),
            ("difficulty", r"Difficulty:\s*(.+)"),
            ("domain", r"Domain:\s*(.+)"),
        ]
        for key, pattern in fields:
            m = re.search(pattern, block)
            if m:
                summary[key] = m.group(1).strip()
        return summary


def solve_single(pdf_path: str, use_nougat: bool = True, use_vlm: bool = True):
    """Run ``STEPSolver`` and emit a human-readable report via the root logger (stdout)."""
    ensure_dirs()
    configure_logging()
    _log.info("")
    _log.info("  " + "=" * 56)
    _log.info("  STEP Pipeline - math problem solver")
    _log.info("  " + "=" * 56)
    _log.info("")
    _log.info(f"  PDF: {pdf_path}")
    _log.info("  " + "-" * 55)

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    result = solver.solve(pdf_path, verbose=True)

    _log.info("\n  " + "=" * 55)
    if "error" in result:
        _log.info(f"  ERROR: {result['error']}")
    else:
        # --- PDF Profiling ---
        _log.info("  PDF PROFILING")
        _log.info(f"  {'-'*55}")
        _log.info(f"  File:            {result['file']}")
        _log.info(f"  Pages:           {result.get('pages', '?')}")
        _log.info(f"  Chars:           {result.get('chars', 0)}")
        _log.info(f"  Category:        {result.get('category', 'unknown')}")
        sec = result.get("secondary_categories") or []
        if sec:
            _log.info(f"  Also signals:    {', '.join(sec)}")
        _log.info(f"  Surface type:    {result.get('surface_type', 'unknown')}")
        domain_label = "surface_integral" if result.get('domain') == "surface_integral" else "general_math"
        _log.info(f"  Domain:          {domain_label}")
        ls = result.get("l5_system")
        if ls and isinstance(ls, dict):
            l5d = ls.get("domain", domain_label)
            l5sec = ls.get("secondary_categories") or []
            sec_txt = f" - hints->{', '.join(l5sec)}" if l5sec else ""
            _log.info(f"  LLM system:      {l5d}{sec_txt}")
        kws = result.get('keywords', [])
        if kws:
            _log.info(f"  Keywords:        {', '.join(kws)}")
        summary_text = result.get('summary', '')
        if summary_text:
            if len(summary_text) > 100:
                _log.info(f"  Summary:         {summary_text[:100]}...")
            else:
                _log.info(f"  Summary:         {summary_text}")

        _log.info("\n  ANSWER")
        _log.info(f"  {'-'*55}")
        _log.info(f"  Source:          {result['source']}")
        _log.info(f"  Final:           {result['final_answer']}")
        if result.get("consensus"):
            _log.info(f"  Consensus:       yes ({result.get('attempts', '?')} runs agreed)")
        elif result.get("attempts", 1) > 1:
            _log.info(f"  Consensus:       no ({result.get('attempts', '?')} runs)")

        llm_sum = result.get('llm_summary', {})
        if llm_sum:
            _log.info("\n  MODEL SUMMARY")
            _log.info(f"  {'-'*55}")
            field_labels = {
                "problem_type": "Problem type",
                "method_used": "Method",
                "surface": "Surface",
                "key_steps": "Key steps",
                "difficulty": "Difficulty",
                "domain": "Domain",
            }
            for key, label in field_labels.items():
                val = llm_sum.get(key)
                if val:
                    _log.info(f"  {label+':':<20} {val}")

        _log.info(f"\n  Total time:      {result['elapsed_s']}s")
    _log.info("  " + "=" * 55)
    _log.info("")
    return result


def solve_batch(pdf_dir: str, count: int = None, use_nougat: bool = True, use_vlm: bool = True):
    """Run every ``*.pdf`` in order; append ``pipeline_log_*.json`` under ``RESULTS_DIR`` via ``PipelineLogger``."""
    ensure_dirs()
    configure_logging()
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        _log.info(f"  [!] No PDFs in {pdf_dir}")
        return

    if count:
        pdfs = pdfs[:count]

    _log.info("")
    _log.info("  " + "=" * 44)
    _log.info("  STEP Pipeline - batch run")
    _log.info("  " + "=" * 44)
    _log.info(f"\n  {len(pdfs)} PDF(s) queued")

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    logger = PipelineLogger(RESULTS_DIR)
    logger.log_config({"ensemble": False, "nougat_dpi": NOUGAT_DPI,
                        "total_pdfs": len(pdfs), "pdf_dir": str(pdf_dir)})
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.log_environment({
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": gpu_name,
    })

    results = []
    ok_count = 0

    for i, pdf in enumerate(pdfs):
        _log.info(f"\n  --- [{i+1}/{len(pdfs)}] {pdf.name} ---")
        logger.start_pdf(pdf.name)
        r = solver.solve(pdf, verbose=True)
        results.append(r)

        fin = "ok" if not r.get("error") else "error"
        if fin == "ok":
            ok_count += 1
        logger.finish_pdf(fin)

    log_path = logger.save()
    logger.print_summary()

    _log.info("\n  " + "=" * 50)
    _log.info(f"  BATCH: {ok_count}/{len(pdfs)} completed without error")
    _log.info(f"  Log: {log_path}")
    _log.info("  " + "=" * 50 + "\n")
    return results


def check_system():
    """Log GPU, API keys, VLM status, and PDF folder count (``python run.py --check``)."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ensure_dirs()
    configure_logging()

    _log.info("")
    _log.info("  " + "=" * 44)
    _log.info("  STEP Pipeline - health check")
    _log.info("  " + "=" * 44)
    _log.info("")

    # GPU
    if torch.cuda.is_available():
        _log.info(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
        _log.info(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        _log.info("  [!] No CUDA GPU (Nougat is slow on CPU)")

    # API Keys
    from config import GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
    keys = [("Groq", GROQ_API_KEY), ("Gemini", GEMINI_API_KEY),
            ("Anthropic", ANTHROPIC_API_KEY), ("OpenAI", OPENAI_API_KEY)]
    for name, key in keys:
        status = "OK" if key else "NO"
        _log.info(f"  [{status}] {name} API key: {'...'+key[-8:] if key else 'missing'}")

    # VLM
    vlm = Layer3_VLM()
    vlm_st = "OK" if vlm.is_available else "NO"
    vlm_name = f"{vlm.provider}/{vlm.model}" if vlm.is_available else "disabled"
    _log.info(f"  [{vlm_st}] VLM: {vlm_name}")

    # PDF count
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    pdf_st = "OK" if pdf_count else "!"
    _log.info(f"  [{pdf_st}] PDF folder: {pdf_count} file(s) ({PDF_DIR})")
    _log.info("")

    # Hints after --check (also shown in Turkish for local users).
    _log.info("  " + "-" * 44)
    _log.info("  Sonraki adimlar / Next steps")
    _log.info("  " + "-" * 44)
    if pdf_count == 0:
        _log.info("  [!] Cozecek PDF yok. Ornek dosyalari su klasore koyun:")
        _log.info(f"      {PDF_DIR.resolve()}")
        _log.info("")
    _log.info("  Tek PDF:     python run.py <dosya.pdf>")
    _log.info("  Klasor:      python run.py Surface_Integration/ -n 5")
    _log.info("  Hizli yol:   python run.py dosya.pdf --no-nougat")
    _log.info("  Web arayuz:  python web_app.py")
    _log.info("               -> tarayici: http://127.0.0.1:5000")
    _log.info("  Eski batch:  python main.py -n 5")
    _log.info("  (Windows'ta garip karakterler icin: $env:PYTHONIOENCODING=\"utf-8\")")
    _log.info("")


def main():
    parser = argparse.ArgumentParser(
        description="STEP — surface-integral PDF solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to a .pdf file or a folder of PDFs")
    parser.add_argument("-n", "--count", type=int, default=None,
                        help="Folder mode: max number of PDFs to process")
    parser.add_argument("--no-nougat", action="store_true",
                        help="Skip Nougat OCR (VLM-only path, often ~3x faster)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip the vision model (Nougat + raw text only)")
    parser.add_argument("--check", action="store_true",
                        help="Print GPU / API key / VLM status")

    args = parser.parse_args()

    if args.check:
        check_system()
        return

    if args.input is None:
        parser.print_help()
        ensure_dirs()
        configure_logging()
        _log.info("\nExamples:")
        _log.info("  python run.py problem.pdf")
        _log.info("  python run.py Surface_Integration/si1.pdf")
        _log.info("  python run.py Surface_Integration/")
        _log.info("  python run.py Surface_Integration/ -n 10")
        _log.info("  python run.py my_problem.pdf --no-nougat")
        _log.info("  python run.py my_problem.pdf --no-vlm")
        _log.info("  python run.py --check")
        return

    input_path = Path(args.input)
    use_nougat = not args.no_nougat
    use_vlm = not args.no_vlm

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        solve_single(str(input_path), use_nougat=use_nougat, use_vlm=use_vlm)
    elif input_path.is_dir():
        solve_batch(str(input_path), count=args.count, use_nougat=use_nougat, use_vlm=use_vlm)
    else:
        ensure_dirs()
        configure_logging()
        _log.info(f"\n  [!] Invalid path: {input_path}")
        _log.info("      Pass a .pdf file or a folder that contains PDFs.\n")


if __name__ == "__main__":
    main()
