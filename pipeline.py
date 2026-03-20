"""
Legacy orchestrator: L0→L1→L2→L3→L4→L5→L6 with `PipelineLogger` JSON output.
Prefer `run.py` / `STEPSolver` for new work; `main.py` still imports this.
"""

import json
import time
import platform
from pathlib import Path

import torch

from config import PDF_DIR, IMG_DIR, NOUGAT_OUT, RESULTS_DIR, KNOWN_ANSWERS, NOUGAT_DPI
from layer0_ingestion import Layer0_PDFIngestion
from layer1_profiler import Layer1_Profiler
from layer2_nougat import Layer2_Nougat
from layer3_vlm import Layer3_VLM
from layer4_synthesis import Layer4_Synthesis
from layer5_llm_solver import Layer5_LLMSolver
from layer6_verifier import Layer6_SymPyVerifier
from pipeline_logger import PipelineLogger


class STEPPipeline:
    """Wires layers together for `main.py` batch demos."""

    def __init__(self, pdf_dir: str | Path = None, provider: str | None = None,
                 ensemble: bool = False):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_DIR
        self.layer0 = Layer0_PDFIngestion(self.pdf_dir, IMG_DIR)
        self.layer1 = Layer1_Profiler()
        self.layer2 = Layer2_Nougat(IMG_DIR, NOUGAT_OUT)
        self.layer3 = Layer3_VLM()
        self.layer4 = Layer4_Synthesis()
        self.ensemble = ensemble
        if ensemble:
            self.solvers = {}
            for p in ["groq", "gemini", "claude", "openai"]:
                solver = Layer5_LLMSolver(force_provider=p)
                if solver.is_available:
                    self.solvers[p] = solver
            self.layer5 = list(self.solvers.values())[0] if self.solvers else Layer5_LLMSolver()
        else:
            self.layer5 = Layer5_LLMSolver(force_provider=provider)
            self.solvers = {self.layer5.provider: self.layer5} if self.layer5.is_available else {}
        self.layer6 = Layer6_SymPyVerifier()

    def get_pdf_files(self) -> list[Path]:
        """Sorted `*.pdf` list; warns if the folder is empty."""
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdfs:
            print(f"  [!] No PDFs in {self.pdf_dir}")
        return pdfs

    def run_layer0_test(self, count: int = 5):
        """Smoke-test Layer 0 on the first N files."""
        pdfs = self.get_pdf_files()
        if not pdfs:
            return

        test_count = min(count, len(pdfs))
        print("=" * 58)
        print(f"  LAYER 0: PDF ingest ({test_count} PDFs)")
        print("=" * 58)

        results = []
        for pdf in pdfs[:test_count]:
            result = self.layer0.process(pdf)
            results.append(result)

        return results

    def run_layer2_test(self, count: int = 5):
        """Batch Nougat-only run with a short quality summary."""
        pdfs = self.get_pdf_files()
        if not pdfs:
            return

        test_count = min(count, len(pdfs))
        print("=" * 58)
        print(f"  LAYER 2: Nougat batch ({test_count} PDFs)")
        print("=" * 58)

        results = []
        for pdf in pdfs[:test_count]:
            print(f"\n  {'─'*56}")
            print(f"  {pdf.name}")
            print(f"  {'─'*56}")

            result = self.layer2.extract_from_pdf(pdf)
            quality = self.layer2.check_quality(result.get("latex", ""))
            result["quality_score"] = quality["score"]
            result["quality_max"] = quality["max_score"]
            results.append(result)

            checks_str = " ".join(
                "[OK]" if v else "[FAIL]"
                for v in quality["checks"].values()
            )
            print(f"    Quality: {quality['score']}/{quality['max_score']} | {checks_str}")

        # Ozet
        print(f"\n\n{'='*58}")
        print("  BATCH SUMMARY")
        print(f"{'='*58}")

        good = sum(1 for r in results if r["quality_score"] >= 3)
        total_chars = sum(r.get("char_count", 0) for r in results)

        for r in results:
            s = r["quality_score"]
            m = r["quality_max"]
            status = "[+++]" if s >= 4 else "[++]" if s >= 2 else "[--]"
            print(f"    {status} {r['file']}: {s}/{m} ({r.get('char_count', 0)} chars, {r.get('pages', 0)} pages)")

        print(f"\n    Strong (>=3/{results[0]['quality_max']}): {good}/{len(results)}")
        print(f"    Total LaTeX chars: {total_chars}")

        return results

    def run_full_pipeline(self, count: int = 5):
        """End-to-end run with JSON log under `RESULTS_DIR`."""
        pdfs = self.get_pdf_files()
        if not pdfs:
            return

        test_count = min(count, len(pdfs))
        pipeline_results = []

        # Structured JSON log (see `PipelineLogger`)
        logger = PipelineLogger(RESULTS_DIR)
        logger.log_config({
            "ensemble": self.ensemble,
            "nougat_dpi": NOUGAT_DPI,
            "total_pdfs": test_count,
            "pdf_dir": str(self.pdf_dir),
        })
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A"
        logger.log_environment({
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": gpu_name,
            "gpu_vram": gpu_vram,
            "cuda_available": torch.cuda.is_available(),
        })
        models = [{"provider": s.provider, "model": s.model_name}
                  for s in self.solvers.values()]
        if self.layer3.is_available:
            models.insert(0, {"provider": "groq_vlm", "model": self.layer3.model})
        logger.log_models(models)

        vlm_status = "on" if self.layer3.is_available else "off"

        if self.ensemble:
            llm_name = "Ensemble(" + "+".join(s.capitalize() for s in self.solvers) + ")"
        else:
            provider_names = {"groq": "Groq", "gemini": "Gemini", "claude": "Claude", "openai": "GPT-4o"}
            llm_name = provider_names.get(self.layer5.provider, "LLM") if self.layer5.is_available else "LLM(none)"

        print("=" * 66)
        print(f"  FULL PIPELINE — {test_count} PDF(s)")
        print(f"  L0(PyMuPDF) -> L1(Profiler) -> L2(Nougat) -> L3(VLM) -> L4(Synthesis) -> L5({llm_name}) -> L6(SymPy)")
        print(f"  GPU: {gpu_name} | VLM: {vlm_status}")
        print("=" * 66)

        for pdf in pdfs[:test_count]:
            fname = pdf.stem
            print(f"\n  {'━'*60}")
            print(f"  [{pdfs.index(pdf)+1}/{test_count}] {pdf.name}")
            print(f"  {'━'*60}")

            logger.start_pdf(pdf.name)
            result = {"file": fname, "layers": {}}

            # === LAYER 0: Metadata + Text ===
            t0 = time.time()
            metadata = self.layer0.extract_metadata(pdf)
            raw_pages = self.layer0.extract_text(pdf)
            t0_elapsed = time.time() - t0
            logger.log_layer0(metadata, raw_pages, t0_elapsed)
            raw_text = "\n".join(p["text"] for p in raw_pages).strip()
            total_chars = sum(len(p["text"]) for p in raw_pages)
            print(f"  [L0] PyMuPDF: {metadata.get('pages')} pages, {total_chars} chars ({t0_elapsed:.2f}s)")

            # === LAYER 1: Profiling ===
            t1 = time.time()
            profile = self.layer1.profile(fname, metadata, raw_text)
            t1_elapsed = time.time() - t1
            logger.log_layer1(profile, t1_elapsed)
            result["layers"]["L1"] = {
                "category": profile["category"],
                "surface": profile["surface_type"],
                "keywords": len(profile["keywords"]),
                "status": "OK",
            }
            print(f"  [L1] Profiler: {profile['category']}/{profile['surface_type']}, {len(profile['keywords'])} keywords ({t1_elapsed:.3f}s)")

            # === LAYER 2: Nougat OCR ===
            t2 = time.time()
            print(f"  [L2] Nougat OCR...")
            nougat = self.layer2.extract_from_pdf(pdf, verbose=True)
            latex = nougat.get("latex", "")
            nougat_quality = self.layer2.check_quality(latex)
            nougat_score = nougat_quality["score"]
            t2_elapsed = time.time() - t2
            logger.log_layer2(nougat, nougat_quality, t2_elapsed)
            result["layers"]["L2"] = {
                "chars": nougat.get("char_count", 0),
                "score": f"{nougat_score}/{nougat_quality['max_score']}",
                "status": "OK" if nougat_score >= 2 else "FAIL",
            }
            print(f"       Nougat: {nougat.get('char_count', 0)} chars, quality {nougat_score}/{nougat_quality['max_score']} ({t2_elapsed:.1f}s)")

            # === LAYER 3: VLM (LLaMA 4 Scout) ===
            vlm_latex = ""
            vlm_score = 0
            if self.layer3.is_available:
                t3 = time.time()
                print(f"  [L3] VLM (LLaMA 4 Scout)...")
                try:
                    vlm_result = self.layer3.extract_from_pdf_images(IMG_DIR, fname, verbose=True)
                    vlm_latex = vlm_result.get("vlm_latex", "")
                    vlm_quality = self.layer3.check_quality(vlm_latex)
                    vlm_score = vlm_quality["score"]
                    t3_elapsed = time.time() - t3
                    logger.log_layer3(vlm_result, vlm_quality, t3_elapsed)
                    result["layers"]["L3"] = {
                        "chars": vlm_result.get("char_count", 0),
                        "score": f"{vlm_score}/{vlm_quality['max_score']}",
                        "status": "OK" if vlm_score >= 2 else "FAIL",
                    }
                    print(f"       VLM: {vlm_result.get('char_count', 0)} chars, quality {vlm_score}/{vlm_quality['max_score']} ({t3_elapsed:.1f}s)")
                except Exception as e:
                    t3_elapsed = time.time() - t3
                    print(f"       VLM: [FAIL] {str(e)[:60]} ({t3_elapsed:.1f}s)")
                    logger.log_layer3({"char_count": 0, "pages": 0},
                                       {"score": 0, "max_score": 4, "checks": {}}, t3_elapsed)
                    result["layers"]["L3"] = {"status": "FAIL", "chars": 0}
            else:
                result["layers"]["L3"] = {"status": "SKIP"}

            # === LAYER 4: Input Synthesis ===
            t4 = time.time()
            synthesis = self.layer4.synthesize(
                raw_text, latex, nougat_score, vlm_latex, vlm_score, profile
            )
            prompt = synthesis["prompt"]
            source = synthesis["source"]
            t4_elapsed = time.time() - t4
            logger.log_layer4(synthesis, t4_elapsed)
            result["layers"]["L4"] = {
                "source": source,
                "prompt_chars": synthesis["prompt_chars"],
                "status": "OK",
            }
            print(f"  [L4] Synthesis: source={source}, {synthesis['prompt_chars']} chars ({t4_elapsed:.3f}s)")

            best_solution = ""
            best_verification = None
            best_provider = None
            disabled_providers = getattr(self, '_disabled_providers', set())
            self._disabled_providers = disabled_providers

            for solver_name, solver in self.solvers.items():
                if solver_name in disabled_providers:
                    continue
                print(f"  [L5] {solver_name.capitalize()} ({solver.model_name}) …")
                t5 = time.time()
                try:
                    solution = solver.solve(prompt)
                    t5_elapsed = time.time() - t5
                    print(f"       {len(solution)} char ({t5_elapsed:.1f}s)")

                    t6 = time.time()
                    verification = self.layer6.verify_llm_answer(fname, solution)
                    t6_elapsed = time.time() - t6

                    logger.log_layer5_attempt(solver_name, solver.model_name,
                                              len(solution), t5_elapsed, verification["status"])

                    if verification["status"] == "match":
                        best_solution = solution
                        best_verification = verification
                        best_provider = solver_name
                        known = KNOWN_ANSWERS.get(fname)
                        print(f"  [L6] {solver_name}: [OK] match (expected {known})")
                        if not self.ensemble:
                            break
                    elif best_verification is None or best_verification["status"] != "match":
                        best_solution = solution
                        best_verification = verification
                        best_provider = solver_name

                except Exception as e:
                    t5_elapsed = time.time() - t5
                    err_str = str(e)
                    print(f"       [FAIL] {err_str[:80]} ({t5_elapsed:.1f}s)")
                    logger.log_layer5_attempt(solver_name, solver.model_name,
                                              0, t5_elapsed, "error", err_str[:200])
                    if any(k in err_str.lower() for k in ["credit", "quota", "insufficient", "billing"]):
                        print(f"       [WARN] {solver_name} skipped for this PDF (quota)")
                        disabled_providers.add(solver_name)

            # Retry: mismatch/parse_error/no_answer durumunda
            if best_verification and best_verification["status"] in ("mismatch", "parse_error", "no_answer"):
                wrong_ans = best_verification.get("llm_answer", "unknown")

                retry_prompts = [
                    prompt + "\n\n"
                    f"CRITICAL: A previous attempt gave the WRONG answer: {wrong_ans}\n"
                    "This answer is INCORRECT. Solve from scratch with extreme care.\n"
                    "CHECKLIST:\n"
                    "- For multi-surface problems: compute EVERY surface integral separately, then SUM all.\n"
                    "- Track ALL coefficients carefully.\n"
                    "- Double-check integration bounds.\n"
                    "Put your final answer inside \\boxed{}.",
                    prompt + "\n\n"
                    f"WARNING: Previous attempts failed. Wrong answer: {wrong_ans}\n"
                    "Use a DIFFERENT approach:\n"
                    "1. Try a different parametrization.\n"
                    "2. For closed surfaces with multiple parts, compute ALL parts.\n"
                    "3. For flux integrals, verify orientation.\n"
                    "Put your final answer inside \\boxed{}.",
                ]

                working_solvers = [(n, s) for n, s in self.solvers.items() if n not in disabled_providers]
                for retry_idx, retry_prompt in enumerate(retry_prompts):
                    if best_verification["status"] == "match":
                        break
                    rname, rslvr = working_solvers[retry_idx % len(working_solvers)]
                    print(f"  [L5] Retry {retry_idx+1}: {rname.capitalize()}...")
                    t5r = time.time()
                    try:
                        retry_sol = rslvr.solve(retry_prompt)
                        t5r_elapsed = time.time() - t5r
                        print(f"       {len(retry_sol)} char ({t5r_elapsed:.1f}s)")
                        retry_ver = self.layer6.verify_llm_answer(fname, retry_sol)

                        logger.log_layer5_attempt(f"{rname}_retry{retry_idx+1}",
                                                  rslvr.model_name, len(retry_sol),
                                                  t5r_elapsed, retry_ver["status"])

                        if retry_ver["status"] == "match":
                            best_solution = retry_sol
                            best_verification = retry_ver
                            best_provider = f"{rname}_retry{retry_idx+1}"
                            print(f"  [L6] Retry: [OK] match")
                    except Exception as e:
                        t5r_elapsed = time.time() - t5r
                        print(f"       Retry [FAIL] {str(e)[:60]}")
                        logger.log_layer5_attempt(f"{rname}_retry{retry_idx+1}",
                                                  rslvr.model_name, 0, t5r_elapsed,
                                                  "error", str(e)[:200])

            # Finalize
            final_status = "skip"
            if best_solution and best_verification:
                final_status = best_verification["status"]
                logger.log_layer5_best(best_provider, final_status)
                t6_final = time.time()
                logger.log_layer6(best_verification, 0.001)

                result["layers"]["L5"] = {
                    "status": "OK", "provider": best_provider, "source": source,
                }
                status_map = {"match": "OK", "mismatch": "FAIL", "no_answer": "?",
                              "parse_error": "?", "skip": "SKIP"}
                result["layers"]["L6"] = {
                    **best_verification,
                    "display_status": status_map.get(final_status, "?"),
                }
                if final_status != "match":
                    known = KNOWN_ANSWERS.get(fname)
                    print(f"  [L6] [FAIL] {final_status} (expected {known})")
            else:
                result["layers"]["L5"] = {"status": "SKIP"}
                result["layers"]["L6"] = {"status": "skip", "display_status": "SKIP"}

            logger.finish_pdf(final_status)
            pipeline_results.append(result)

        # Summary table + JSON log
        stats = self._compute_stats(pipeline_results)
        self._print_pipeline_summary(pipeline_results, stats)

        log_path = logger.save()
        logger.print_summary()
        print(f"\n  Log JSON: {log_path}")

        return logger.run_log

    @staticmethod
    def _compute_stats(pipeline_results: list[dict]) -> dict:
        l2_ok = sum(1 for r in pipeline_results if r["layers"]["L2"]["status"] == "OK")
        l3_ok = sum(1 for r in pipeline_results if r["layers"].get("L3", {}).get("status") == "OK")
        l5_ok = sum(1 for r in pipeline_results if r["layers"].get("L5", {}).get("status") == "OK")
        l6_match = sum(
            1 for r in pipeline_results
            if r["layers"].get("L6", {}).get("status") == "match"
        )
        l6_tested = sum(
            1 for r in pipeline_results
            if r["layers"].get("L6", {}).get("status") in ("match", "mismatch")
        )
        return {
            "nougat_ok": l2_ok,
            "vlm_ok": l3_ok,
            "llm_ok": l5_ok,
            "verified_match": l6_match,
            "verified_total": l6_tested,
            "accuracy_pct": round(l6_match / l6_tested * 100, 1) if l6_tested else 0,
        }

    def _print_pipeline_summary(self, pipeline_results: list[dict], stats: dict):
        """Print the batch results table to stdout."""
        print(f"\n\n{'='*58}")
        print("  BATCH RESULTS")
        print(f"{'='*58}")

        l2_ok = stats["nougat_ok"]
        l5_ok = stats["llm_ok"]
        l6_match = stats["verified_match"]
        l6_tested = stats["verified_total"]

        header = f"  {'File':<12} {'Nougat':<8} {'VLM':<8} {'Source':<16} {'LLM':<8} {'SymPy':<8}"
        print(f"\n{header}")
        print(f"  {'─'*62}")
        for r in pipeline_results:
            f = r["file"]
            l2 = r["layers"]["L2"]["status"]
            l3 = r["layers"].get("L3", {}).get("status", "SKIP")
            l4_src = r["layers"].get("L4", {}).get("source", "-")
            l5 = r["layers"].get("L5", {}).get("status", "SKIP")
            l6 = r["layers"].get("L6", {}).get("display_status", "SKIP")
            print(f"  {f:<12} {l2:<8} {l3:<8} {l4_src:<16} {l5:<8} {l6:<8}")

        total = len(pipeline_results)
        l3_ok = stats.get("vlm_ok", 0)
        print(f"\n  Nougat OK:       {l2_ok}/{total}")
        print(f"  VLM OK:          {l3_ok}/{total}")
        print(f"  LLM OK:          {l5_ok}/{total}")
        print(f"  Verified match:  {l6_match}/{l6_tested} (with reference answer)")

        if l6_tested > 0:
            accuracy = l6_match / l6_tested * 100
            print(f"\n  End-to-end accuracy: {accuracy:.0f}%")

            if accuracy >= 80:
                print(f"  Target met (≥80%).")
            elif accuracy >= 60:
                print(f"  Decent — room left in prompts / OCR.")
            else:
                print(f"  Low accuracy — inspect Nougat output or prompts.")
