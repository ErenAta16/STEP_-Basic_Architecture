"""
STEP pipeline — main CLI entry.

Usage:
    python run.py problem.pdf                    # single PDF
    python run.py problem.pdf --no-nougat        # skip Nougat (VLM path, faster)
    python run.py Surface_Integration/           # every PDF in folder
    python run.py Surface_Integration/ -n 5       # first 5 only
    python run.py --check                        # GPU / keys / VLM sanity check
    python run.py --benchmark Surface_Integration/ --no-nougat   # accuracy + timings JSON
"""

import argparse
import json
import sys
import time
import platform
from datetime import datetime
from pathlib import Path

import torch

from config import (PDF_DIR, IMG_DIR, NOUGAT_OUT, RESULTS_DIR, KNOWN_ANSWERS, NOUGAT_DPI,
                    LLM_SYSTEM_PROMPT_SURFACE, LLM_SYSTEM_PROMPT_GENERAL)
from layer0_ingestion import Layer0_PDFIngestion
from layer1_profiler import Layer1_Profiler
from layer2_nougat import Layer2_Nougat
from layer3_vlm import Layer3_VLM
from layer4_synthesis import Layer4_Synthesis
from layer5_llm_solver import Layer5_LLMSolver
from layer6_verifier import Layer6_SymPyVerifier
from pipeline_logger import PipelineLogger


class STEPSolver:
    """Glue L0-L6: ingest -> profile -> optional Nougat/VLM -> LLM -> extract answer -> verify."""

    def __init__(self, use_nougat: bool = True, use_vlm: bool = True):
        self.l0 = Layer0_PDFIngestion(PDF_DIR, IMG_DIR)
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

    def solve(self, pdf_path: str | Path, verbose: bool = True) -> dict:
        """Run the stack on one file; returned dict always includes timings."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        fname = pdf_path.stem
        result = {"file": fname, "pdf_path": str(pdf_path)}
        t_start = time.time()
        timings = {}

        # --- L0: PDF Ingestion ---
        t0 = time.time()
        metadata = self.l0.extract_metadata(pdf_path)
        raw_pages = self.l0.extract_text(pdf_path)
        self.l0.extract_images(pdf_path, dpi=NOUGAT_DPI)
        raw_text = "\n".join(p["text"] for p in raw_pages).strip()
        result["pages"] = metadata.get("pages", 0)
        result["chars"] = sum(len(p["text"]) for p in raw_pages)
        timings["l0_ingest"] = round(time.time() - t0, 3)
        if verbose:
            print(f"  [L0] {metadata.get('pages')} pages, {result['chars']} chars ({timings['l0_ingest']:.2f}s)")

        # --- L1: quick tags from raw text (cheap classifier) ---
        t1 = time.time()
        profile = self.l1.profile(fname, metadata, raw_text)
        timings["l1_profile"] = round(time.time() - t1, 4)
        if verbose:
            print(f"  [L1] {profile['category']} / {profile['surface_type']} ({timings['l1_profile']:.3f}s)")

        # --- L2: Nougat OCR (optional, heavy) ---
        nougat_latex = ""
        nougat_score = 0
        timings["l2_nougat"] = 0.0
        if self.use_nougat and self.l2 is not None:
            t2 = time.time()
            if verbose:
                print(f"  [L2] Nougat OCR...")
            nougat_result = self.l2.extract_from_pdf(pdf_path, verbose=verbose)
            nougat_latex = nougat_result.get("latex", "")
            nougat_quality = self.l2.check_quality(nougat_latex)
            nougat_score = nougat_quality["score"]
            timings["l2_nougat"] = round(time.time() - t2, 2)
            if verbose:
                print(f"       {nougat_result.get('char_count', 0)} chars, quality {nougat_score}/4 ({timings['l2_nougat']:.1f}s)")

        # --- L3: VLM ---
        vlm_latex = ""
        vlm_score = 0
        timings["l3_vlm"] = 0.0
        if self.l3 is not None and self.l3.is_available:
            t3 = time.time()
            if verbose:
                print(f"  [L3] VLM (LLaMA 4 Scout)...")
            try:
                vlm_result = self.l3.extract_from_pdf_images(IMG_DIR, fname, verbose=verbose)
                vlm_latex = vlm_result.get("vlm_latex", "")
                vlm_quality = self.l3.check_quality(vlm_latex)
                vlm_score = vlm_quality["score"]
                timings["l3_vlm"] = round(time.time() - t3, 2)
                if verbose:
                    print(f"       {vlm_result.get('char_count', 0)} chars, quality {vlm_score}/4 ({timings['l3_vlm']:.1f}s)")
            except Exception as e:
                timings["l3_vlm"] = round(time.time() - t3, 2)
                if verbose:
                    print(f"       [FAIL] {str(e)[:60]}")

        # --- L1b: Re-profile with OCR/VLM text if raw was empty ---
        if result["chars"] == 0 and (vlm_latex or nougat_latex):
            ocr_text = vlm_latex or nougat_latex
            profile = self.l1.profile(fname, metadata, raw_text, latex_text=ocr_text)
            if verbose:
                print(f"  [L1b] Re-profiled: {profile['category']} / {profile['surface_type']}")

        result["category"] = profile["category"]
        result["surface_type"] = profile["surface_type"]
        result["keywords"] = profile["keywords"]
        result["summary"] = profile["summary"]

        # --- L4: Synthesis ---
        t4 = time.time()
        synthesis = self.l4.synthesize(raw_text, nougat_latex, nougat_score,
                                        vlm_latex, vlm_score, profile)
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
            print(f"  [L4] source={synthesis['source']}, domain={domain_label}, prompt_chars={synthesis['prompt_chars']}")

        # --- L5: LLM Solve with Consensus (2-3 attempts) ---
        system_prompt = LLM_SYSTEM_PROMPT_SURFACE if domain == "surface_integral" else LLM_SYSTEM_PROMPT_GENERAL
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

        # --- L6: compare to KNOWN_ANSWERS when we have one ---
        t6 = time.time()
        known = KNOWN_ANSWERS.get(fname)
        if known is not None:
            verification = self.l6.verify_llm_answer(fname, best["solution"])
            result["verification"] = verification["status"]
            result["known_answer"] = str(known)
            if verbose:
                tag = "[OK]" if verification["status"] == "match" else "[FAIL]"
                print(f"  [L6] {tag} {verification['status']} (expected {known})")
        else:
            result["verification"] = "no_known_answer"
            if verbose:
                print(f"  [L6] No reference answer; skipped numeric check")

        # --- strip FINAL_ANSWER / boxed from model text ---
        fa = self.l6._extract_final_answer(best["solution"])
        result["final_answer"] = fa if fa else "(could not extract)"

        # --- optional SUMMARY: block for the UI ---
        result["llm_summary"] = self._extract_llm_summary(best["solution"])

        timings["l6_verify"] = round(time.time() - t6, 4)
        result["timings"] = timings
        result["elapsed_s"] = round(time.time() - t_start, 1)
        return result

    def _solve_with_consensus(self, prompt: str, system_prompt: str,
                               max_attempts: int = 3, verbose: bool = True) -> list[dict]:
        """Multi-model consensus with rate-limit delay and 503 retry."""
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
                    print(f"  [L5] All models busy; sleeping {RETRY_503_DELAY}s...")
                time.sleep(RETRY_503_DELAY)
                rate_limited.clear()
                consecutive_503 = 0
                solver = solvers[0][1]
                solver_label = f"{solver.provider}/{solver.model_name}"

            t5 = time.time()
            if verbose:
                print(f"  [L5] {solver_label} [{i+1}/{max_attempts}]...", end=" ")
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
                    print(f"-> {fa or '?'} ({elapsed:.1f}s)")

                if i > 0 and fa_key is not None and answer_counts.get(fa_key, 0) >= 2:
                    if verbose:
                        print(f"  [L5] [OK] Consensus: {fa} ({answer_counts[fa_key]}/{i+1} agree)")
                    for a in attempts:
                        if a["key"] == fa_key:
                            a["consensus"] = True
                    return [a for a in attempts if a["key"] == fa_key][:1]

            except Exception as e:
                err = str(e)
                if verbose:
                    print(f"[ERR] {err[:80]}")

                is_transient = any(k in err for k in ["503", "UNAVAILABLE", "overloaded", "high demand"])
                is_rate = any(k in err for k in ["429", "RESOURCE_EXHAUSTED", "rate", "quota"])
                is_fatal = any(k in err for k in ["404", "NOT_FOUND"])

                if is_transient and consecutive_503 < 2:
                    consecutive_503 += 1
                    delay = RETRY_503_DELAY * consecutive_503
                    if verbose:
                        print(f"  [L5] [RETRY] HTTP 503: same model again in {delay}s...")
                    time.sleep(delay)
                elif is_rate or is_fatal or (is_transient and consecutive_503 >= 2):
                    consecutive_503 = 0
                    for tag, s in solvers:
                        if s is solver:
                            rate_limited.add(tag)
                            if verbose:
                                print(f"  [L5] [WARN] {solver_label}: switching to fallback")
                            break

                attempts.append({"solution": "", "final_answer": "", "error": err})

        if not attempts:
            return []

        valid_counts = {k: v for k, v in answer_counts.items() if k is not None}
        if valid_counts:
            best_key = max(valid_counts, key=valid_counts.get)
            best_count = valid_counts[best_key]
            if verbose and best_count < 2:
                print(f"  [L5] [WARN] Weak consensus; best guess: {best_key} ({best_count}/{len(attempts)})")
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
        """Pull key/value lines after `SUMMARY:` if the model followed the template."""
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
    """CLI pretty-printer around `STEPSolver.solve`."""
    print()
    print("  " + "=" * 56)
    print("  STEP Pipeline — math problem solver")
    print("  " + "=" * 56)
    print()
    print(f"  PDF: {pdf_path}")
    print("  " + "-" * 55)

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    result = solver.solve(pdf_path, verbose=True)

    print("\n  " + "=" * 55)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        # --- PDF Profiling ---
        print(f"  PDF PROFILING")
        print(f"  {'─'*55}")
        print(f"  File:            {result['file']}")
        print(f"  Pages:           {result.get('pages', '?')}")
        print(f"  Chars:           {result.get('chars', 0)}")
        print(f"  Category:        {result.get('category', 'unknown')}")
        print(f"  Surface type:    {result.get('surface_type', 'unknown')}")
        domain_label = "surface_integral" if result.get('domain') == "surface_integral" else "general_math"
        print(f"  Domain:          {domain_label}")
        kws = result.get('keywords', [])
        if kws:
            print(f"  Keywords:        {', '.join(kws)}")
        summary_text = result.get('summary', '')
        if summary_text:
            if len(summary_text) > 100:
                print(f"  Summary:         {summary_text[:100]}...")
            else:
                print(f"  Summary:         {summary_text}")

        print(f"\n  ANSWER")
        print(f"  {'─'*55}")
        print(f"  Source:          {result['source']}")
        print(f"  Final:           {result['final_answer']}")
        if result.get("consensus"):
            print(f"  Consensus:       yes ({result.get('attempts', '?')} runs agreed)")
        elif result.get("attempts", 1) > 1:
            print(f"  Consensus:       no ({result.get('attempts', '?')} runs)")

        if result["verification"] == "match":
            print(f"  Verification:    MATCH (expected {result.get('known_answer', '?')})")
        elif result["verification"] == "no_known_answer":
            print(f"  Verification:    n/a (no benchmark answer)")
        else:
            print(f"  Verification:    {result['verification']}")

        llm_sum = result.get('llm_summary', {})
        if llm_sum:
            print(f"\n  MODEL SUMMARY")
            print(f"  {'─'*55}")
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
                    print(f"  {label+':':<20} {val}")

        print(f"\n  Total time:      {result['elapsed_s']}s")
    print("  " + "=" * 55)
    print()
    return result


def solve_batch(pdf_dir: str, count: int = None, use_nougat: bool = True, use_vlm: bool = True):
    """Process every `*.pdf` in a directory (logs go to `RESULTS_DIR`)."""
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  [!] No PDFs in {pdf_dir}")
        return

    if count:
        pdfs = pdfs[:count]

    print()
    print("  " + "=" * 44)
    print("  STEP Pipeline — batch run")
    print("  " + "=" * 44)
    print(f"\n  {len(pdfs)} PDF(s) queued")

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
    match_count = 0
    total_testable = 0

    for i, pdf in enumerate(pdfs):
        fname = pdf.stem
        print(f"\n  --- [{i+1}/{len(pdfs)}] {pdf.name} ---")
        logger.start_pdf(pdf.name)
        r = solver.solve(pdf, verbose=True)
        results.append(r)

        if r.get("verification") == "match":
            match_count += 1
            total_testable += 1
        elif r.get("verification") in ("mismatch", "parse_error"):
            total_testable += 1

        logger.finish_pdf(r.get("verification", "skip"))

    log_path = logger.save()
    logger.print_summary()

    print("\n  " + "=" * 50)
    print(f"  BATCH: {match_count}/{total_testable} matched reference answers")
    print(f"  Log: {log_path}")
    print("  " + "=" * 50 + "\n")
    return results


def run_benchmark(
    pdf_dir: str,
    count: int | None = None,
    use_nougat: bool = True,
    use_vlm: bool = True,
):
    """Accuracy vs `KNOWN_ANSWERS` plus mean layer timings; writes JSON under `RESULTS_DIR`."""
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  [!] No PDFs under {pdf_dir}.")
        return None

    if count:
        pdfs = pdfs[:count]

    print()
    print("  " + "=" * 62)
    print("  STEP — performance / benchmark")
    print("  " + "=" * 62)
    print(f"\n  Folder:           {pdf_dir.resolve()}")
    print(f"  PDF count:        {len(pdfs)}")
    print(f"  Nougat:           {'on' if use_nougat else 'off'}")
    print(f"  VLM:              {'on' if use_vlm else 'off'}")
    print(f"  Reference set:    {len(KNOWN_ANSWERS)} problems")
    print()

    solver = STEPSolver(use_nougat=use_nougat, use_vlm=use_vlm)
    rows = []
    status_counts: dict[str, int] = {}
    sum_t = {k: 0.0 for k in ("l0_ingest", "l1_profile", "l2_nougat", "l3_vlm",
                              "l4_synthesis", "l5_llm", "l6_verify", "total")}

    for i, pdf in enumerate(pdfs):
        fname = pdf.stem
        print(f"  [{i+1:3d}/{len(pdfs)}] {fname} ... ", end="", flush=True)
        r = solver.solve(pdf, verbose=False)

        if r.get("error"):
            st = "error"
            print(f"ERR: {r['error'][:50]}")
        else:
            st = r.get("verification", "unknown")
            ok = st == "match"
            tag = "OK" if ok else "FAIL"
            print(f"{tag}  {st}  ({r.get('elapsed_s', 0)}s)")

        status_counts[st] = status_counts.get(st, 0) + 1

        tim = r.get("timings") or {}
        for k in sum_t:
            if k == "total":
                sum_t[k] += float(r.get("elapsed_s") or 0)
            elif k in tim:
                sum_t[k] += float(tim[k])

        rows.append({
            "file": fname,
            "verification": r.get("verification"),
            "source": r.get("source"),
            "domain": r.get("domain"),
            "nougat_score": r.get("nougat_score"),
            "vlm_score": r.get("vlm_score"),
            "prompt_chars": r.get("prompt_chars"),
            "attempts": r.get("attempts"),
            "consensus": r.get("consensus"),
            "model_used": r.get("model_used"),
            "final_answer": r.get("final_answer"),
            "elapsed_s": r.get("elapsed_s"),
            "timings": tim,
            "error": r.get("error"),
        })

    n = len(pdfs)
    testable = sum(status_counts.get(s, 0) for s in ("match", "mismatch", "parse_error", "no_answer"))
    matches = status_counts.get("match", 0)
    acc = (100.0 * matches / testable) if testable else 0.0

    def avg(key: str) -> float:
        return sum_t[key] / n if n else 0.0

    print()
    print("  " + "─" * 58)
    print("  SUMMARY — verification (problems with a reference answer)")
    print("  " + "─" * 58)
    for label, key in [
        ("match", "match"),
        ("mismatch", "mismatch"),
        ("parse_error", "parse_error"),
        ("no_answer", "no_answer"),
        ("no_known_answer", "no_known_answer"),
        ("error (pipeline)", "error"),
    ]:
        c = status_counts.get(key, 0)
        if c or key in ("match", "mismatch"):
            print(f"    {label:36s} {c}")

    print(f"\n  Accuracy (testable): {matches}/{testable} = {acc:.1f}%")
    print()
    print("  " + "─" * 58)
    print("  SUMMARY — mean time (seconds / PDF)")
    print("  " + "─" * 58)
    for key, label in [
        ("l0_ingest", "L0 PDF ingest"),
        ("l1_profile", "L1 profiling"),
        ("l2_nougat", "L2 Nougat"),
        ("l3_vlm", "L3 VLM"),
        ("l4_synthesis", "L4 synthesis"),
        ("l5_llm", "L5 LLM (+ retries)"),
        ("l6_verify", "L6 verify + extract"),
        ("total", "Total"),
    ]:
        print(f"    {label:22s} {avg(key):.2f}s")
    print()

    out = {
        "run_at": datetime.now().isoformat(),
        "pdf_dir": str(pdf_dir.resolve()),
        "count": n,
        "use_nougat": use_nougat,
        "use_vlm": use_vlm,
        "summary": {
            "match": matches,
            "testable": testable,
            "accuracy_pct": round(acc, 2),
            "status_counts": status_counts,
            "avg_timings_s": {k: round(avg(k), 3) for k in sum_t},
        },
        "per_pdf": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  JSON report: {log_path}\n")

    return out


def check_system():
    """Quick environment snapshot before a long run."""
    print()
    print("  " + "=" * 44)
    print("  STEP Pipeline — health check")
    print("  " + "=" * 44)
    print()

    # GPU
    if torch.cuda.is_available():
        print(f"  [OK] GPU: {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print(f"  [!] No CUDA GPU (Nougat is slow on CPU)")

    # API Keys
    from config import GROQ_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
    keys = [("Groq", GROQ_API_KEY), ("Gemini", GEMINI_API_KEY),
            ("Anthropic", ANTHROPIC_API_KEY), ("OpenAI", OPENAI_API_KEY)]
    for name, key in keys:
        status = "OK" if key else "NO"
        print(f"  [{status}] {name} API key: {'...'+key[-8:] if key else 'missing'}")

    # VLM
    vlm = Layer3_VLM()
    vlm_st = "OK" if vlm.is_available else "NO"
    print(f"  [{vlm_st}] VLM (LLaMA 4 Scout): {'ready' if vlm.is_available else 'disabled'}")

    # PDF count
    pdf_count = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    pdf_st = "OK" if pdf_count else "!"
    print(f"  [{pdf_st}] PDF folder: {pdf_count} file(s) ({PDF_DIR})")
    print(f"  [OK] Reference answers: {len(KNOWN_ANSWERS)} problems")
    print()


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
    parser.add_argument("--benchmark", action="store_true",
                        help="Measure accuracy + layer timings; writes JSON under results/")

    args = parser.parse_args()

    if args.check:
        check_system()
        return

    if args.benchmark:
        bench_dir = args.input or str(PDF_DIR)
        p = Path(bench_dir)
        if not p.is_dir():
            print(f"\n  [!] Benchmark needs a directory: {p}")
            print("      Example: python run.py --benchmark Surface_Integration/ --no-nougat\n")
            return
        run_benchmark(
            str(p.resolve()),
            count=args.count,
            use_nougat=not args.no_nougat,
            use_vlm=not args.no_vlm,
        )
        return

    if args.input is None:
        parser.print_help()
        print("\nExamples:")
        print("  python run.py problem.pdf")
        print("  python run.py Surface_Integration/si1.pdf")
        print("  python run.py Surface_Integration/")
        print("  python run.py Surface_Integration/ -n 10")
        print("  python run.py my_problem.pdf --no-nougat")
        print("  python run.py my_problem.pdf --no-vlm")
        print("  python run.py --check")
        print("  python run.py --benchmark Surface_Integration/ --no-nougat -n 10")
        return

    input_path = Path(args.input)
    use_nougat = not args.no_nougat
    use_vlm = not args.no_vlm

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        solve_single(str(input_path), use_nougat=use_nougat, use_vlm=use_vlm)
    elif input_path.is_dir():
        solve_batch(str(input_path), count=args.count, use_nougat=use_nougat, use_vlm=use_vlm)
    else:
        print(f"\n  [!] Invalid path: {input_path}")
        print(f"      Pass a .pdf file or a folder that contains PDFs.\n")


if __name__ == "__main__":
    main()
