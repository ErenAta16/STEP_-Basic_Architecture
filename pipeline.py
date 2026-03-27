"""
Batch driver used by ``main.py``: same stages as ``run.STEPSolver``, plus JSON logging.

For single files or day-to-day use, ``python run.py …`` is usually simpler; this class
exists for multi-PDF runs and structured ``pipeline_log_*.json`` output.
"""

import json
import logging
import time
import platform
from pathlib import Path

import torch

from config import (
    PDF_DIR,
    IMG_DIR,
    NOUGAT_OUT,
    RESULTS_DIR,
    NOUGAT_DPI,
    get_system_prompt,
    ensure_dirs,
)
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


class STEPPipeline:
    """Builds layer objects once, then runs ``run_full_pipeline`` over a PDF directory."""

    def __init__(self, pdf_dir: str | Path = None, provider: str | None = None,
                 ensemble: bool = False, use_nougat: bool = False, use_vlm: bool = True):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_DIR
        self.use_nougat = use_nougat
        self.use_vlm = use_vlm
        self.layer0 = Layer0_PDFIngestion(IMG_DIR)
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
        """Return PDFs sorted by name; logs a warning when the folder has none."""
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdfs:
            _log.info(f"  [!] No PDFs in {self.pdf_dir}")
        return pdfs

    def run_full_pipeline(self, count: int = 5):
        """Process up to ``count`` PDFs and write one JSON log under ``RESULTS_DIR``.

        Ensures pipeline directories exist and configures logging before the run.
        """
        ensure_dirs()
        configure_logging()
        pdfs = self.get_pdf_files()
        if not pdfs:
            return

        test_count = min(count, len(pdfs))
        pipeline_results = []

        logger = PipelineLogger(RESULTS_DIR)
        logger.log_config({
            "ensemble": self.ensemble,
            "nougat_dpi": NOUGAT_DPI,
            "total_pdfs": test_count,
            "pdf_dir": str(self.pdf_dir),
            "use_nougat": self.use_nougat,
            "use_vlm": self.use_vlm,
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
        if self.use_vlm and self.layer3.is_available:
            vp = (self.layer3.provider or "groq") + "_vlm"
            models.insert(0, {"provider": vp, "model": self.layer3.model})
        logger.log_models(models)

        vlm_status = "on" if (self.use_vlm and self.layer3.is_available) else "off"

        if self.ensemble:
            llm_name = "Ensemble(" + "+".join(s.capitalize() for s in self.solvers) + ")"
        else:
            provider_names = {"groq": "Groq", "gemini": "Gemini", "claude": "Claude", "openai": "GPT-4o"}
            llm_name = provider_names.get(self.layer5.provider, "LLM") if self.layer5.is_available else "LLM(none)"

        want_l2 = self.use_nougat
        want_l3 = self.use_vlm and self.layer3.is_available
        if want_l2 and want_l3:
            ocr_desc = "L2+L3 (parallel)"
        elif want_l2:
            ocr_desc = "L2 only"
        elif want_l3:
            ocr_desc = "L3 only"
        else:
            ocr_desc = "no OCR"
        _log.info("=" * 66)
        _log.info(f"  FULL PIPELINE — {test_count} PDF(s)")
        _log.info(
            f"  L0(PyMuPDF+PNG) -> L1 -> {ocr_desc} -> L4 -> L5({llm_name}) -> L6(extract)"
        )
        _log.info(
            f"  GPU: {gpu_name} | Nougat: {'on' if self.use_nougat else 'off'} | VLM: {vlm_status}"
        )
        _log.info("=" * 66)

        for pdf in pdfs[:test_count]:
            fname = pdf.stem
            _log.info(f"\n  {'━'*60}")
            _log.info(f"  [{pdfs.index(pdf)+1}/{test_count}] {pdf.name}")
            _log.info(f"  {'━'*60}")

            logger.start_pdf(pdf.name)
            result = {"file": fname, "layers": {}}

            # === LAYER 0: Metadata + Text + PNGs (align with STEPSolver) ===
            t0 = time.time()
            metadata, raw_pages, _img_meta = self.layer0.extract_metadata_text_and_images(
                pdf, dpi=NOUGAT_DPI
            )
            raw_text = "\n".join(p["text"] for p in raw_pages).strip()
            md_text = self.layer0.extract_markdown(pdf, text_pages=raw_pages)
            text_quality = self.layer0.analyze_text_quality(raw_pages)
            t0_elapsed = time.time() - t0
            logger.log_layer0(metadata, raw_pages, t0_elapsed, text_quality=text_quality)
            total_chars = sum(len(p["text"]) for p in raw_pages)
            _log.info(
                f"  [L0] PyMuPDF: {metadata.get('pages')} pages, {total_chars} chars, "
                f"quality {text_quality['score']}/{text_quality['max_score']} ({t0_elapsed:.2f}s)"
            )

            # === LAYER 1: Profiling ===
            t1 = time.time()
            profile = self.layer1.profile(fname, metadata, raw_text)
            t1_elapsed = time.time() - t1
            logger.log_layer1(profile, t1_elapsed)
            result["layers"]["L1"] = {
                "category": profile["category"],
                "secondary_categories": profile.get("secondary_categories", []),
                "surface": profile["surface_type"],
                "keywords": len(profile["keywords"]),
                "status": "OK",
            }
            sec = profile.get("secondary_categories") or []
            sec_s = f" | also: {', '.join(sec)}" if sec else ""
            _log.info(
                f"  [L1] Profiler: {profile['category']}/{profile['surface_type']}{sec_s}, "
                f"{len(profile['keywords'])} keywords ({t1_elapsed:.3f}s)"
            )

            # === LAYER 2 + 3: Nougat + VLM (``parallel_ocr.run_parallel_nougat_vlm``) ===
            latex = ""
            nougat_score = 0
            nougat: dict = {}
            vlm_latex = ""
            vlm_score = 0
            vlm_result: dict = {}
            t2_elapsed = 0.0
            t3_elapsed = 0.0
            nougat_pkg = None
            vlm_pkg = None

            ocr = run_parallel_nougat_vlm(
                pdf_path=pdf,
                fname=fname,
                img_dir=IMG_DIR,
                nougat_layer=self.layer2 if self.use_nougat else None,
                vlm_layer=self.layer3 if self.use_vlm else None,
                use_nougat=self.use_nougat,
                use_vlm=self.use_vlm,
                vlm_available=self.layer3.is_available,
                text_quality=text_quality,
                total_chars=total_chars,
                verbose=True,
                nougat_verbose=True,
            )
            nougat_needed = ocr["nougat_needed"]
            nougat_pkg = ocr["nougat_pkg"]
            vlm_pkg = ocr["vlm_pkg"]
            t2_elapsed = ocr["t2_elapsed"]
            t3_elapsed = ocr["t3_elapsed"]

            if not nougat_needed:
                stub_q = self.layer2.check_quality("")
                stub_n = {"latex": "", "char_count": 0, "pages": 0, "output_path": ""}
                logger.log_layer2(stub_n, stub_q, 0.0, skipped=True)
                latex, nougat_score = "", 0
                nougat = stub_n
                nougat_quality = stub_q
                result["layers"]["L2"] = {
                    "chars": 0,
                    "score": f"{nougat_score}/{stub_q['max_score']}",
                    "status": "SKIP",
                }
            elif nougat_pkg:
                nougat, nougat_quality, t2_elapsed = nougat_pkg
                latex = nougat.get("latex", "")
                nougat_score = nougat_quality["score"]
                logger.log_layer2(nougat, nougat_quality, t2_elapsed)
                result["layers"]["L2"] = {
                    "chars": nougat.get("char_count", 0),
                    "score": f"{nougat_score}/{nougat_quality['max_score']}",
                    "status": "OK" if nougat_score >= 2 else "FAIL",
                }
            else:
                fail_q = self.layer2.check_quality("")
                stub_n = {"latex": "", "char_count": 0, "pages": 0, "output_path": ""}
                logger.log_layer2(stub_n, fail_q, t2_elapsed)
                latex, nougat_score = "", 0
                nougat = stub_n
                nougat_quality = fail_q
                result["layers"]["L2"] = {
                    "chars": 0,
                    "score": f"{nougat_score}/{fail_q['max_score']}",
                    "status": "FAIL",
                }

            vlm_tech = "VLM"
            if self.layer3.is_available:
                vlm_tech = (
                    f"Gemini ({self.layer3.model})"
                    if self.layer3.provider == "gemini"
                    else f"Groq VLM ({self.layer3.model})"
                )

            if not self.use_vlm or not self.layer3.is_available:
                result["layers"]["L3"] = {"status": "SKIP"}
            elif vlm_pkg:
                vlm_result, vlm_quality, t3_elapsed = vlm_pkg
                vlm_latex = vlm_result.get("vlm_latex", "")
                vlm_score = vlm_quality["score"]
                logger.log_layer3(vlm_result, vlm_quality, t3_elapsed, technology=vlm_tech)
                result["layers"]["L3"] = {
                    "chars": vlm_result.get("char_count", 0),
                    "score": f"{vlm_score}/{vlm_quality['max_score']}",
                    "status": "OK" if vlm_score >= 2 else "FAIL",
                }
            else:
                vlm_fail_q = self.layer3.check_quality("")
                logger.log_layer3(
                    {"char_count": 0, "pages": 0},
                    vlm_fail_q,
                    t3_elapsed,
                    technology=vlm_tech,
                )
                result["layers"]["L3"] = {"status": "FAIL", "chars": 0}

            if total_chars == 0 and (vlm_latex or latex):
                t1b = time.time()
                ocr_text = vlm_latex or latex
                profile = self.layer1.profile(fname, metadata, raw_text, latex_text=ocr_text)
                t1b_elapsed = time.time() - t1b
                logger.log_layer1(profile, t1b_elapsed)
                result["layers"]["L1"].update({
                    "category": profile["category"],
                    "secondary_categories": profile.get("secondary_categories", []),
                    "surface": profile["surface_type"],
                    "keywords": len(profile["keywords"]),
                })
                _log.info(
                    f"  [L1b] Re-profiled: {profile['category']} / {profile['surface_type']} "
                    f"({t1b_elapsed:.3f}s)"
                )

            # === LAYER 4: Input Synthesis ===
            t4 = time.time()
            synthesis = self.layer4.synthesize(
                raw_text, latex, nougat_score, vlm_latex, vlm_score, profile,
                md_text=md_text,
            )
            prompt = synthesis["prompt"]
            source = synthesis["source"]
            domain = synthesis.get("domain", "general_math")
            l5_system = {
                "domain": domain,
                "secondary_categories": list(profile.get("secondary_categories") or []),
            }
            system_prompt = get_system_prompt(
                domain,
                secondary_categories=profile.get("secondary_categories"),
                primary_category=profile.get("category"),
            )
            t4_elapsed = time.time() - t4
            logger.log_layer4(synthesis, t4_elapsed, l5_system=l5_system)
            result["l5_system"] = l5_system
            result["layers"]["L4"] = {
                "source": source,
                "domain": domain,
                "prompt_chars": synthesis["prompt_chars"],
                "status": "OK",
            }
            dom_lbl = "surface_integral" if domain == "surface_integral" else "general_math"
            _log.info(
                f"  [L4] Synthesis: source={source}, domain={dom_lbl}, "
                f"{synthesis['prompt_chars']} chars ({t4_elapsed:.3f}s)"
            )
            dom_tag = "surface" if domain == "surface_integral" else "general"
            sec_l5 = profile.get("secondary_categories") or []
            sec_l5_s = f", signals={sec_l5}" if sec_l5 else ""
            _log.info(f"  [L5] system={dom_tag}{sec_l5_s} (per attempt below)")

            best_solution = ""
            best_provider = None
            disabled_providers = getattr(self, '_disabled_providers', set())
            self._disabled_providers = disabled_providers

            # Without ``ensemble``, stop at the first non-empty reply. With ``ensemble``, still hit
            # every provider (useful when comparing backends); ``best_solution`` stays the first hit.
            for solver_name, solver in self.solvers.items():
                if solver_name in disabled_providers:
                    continue
                _log.info(f"  [L5] {solver_name.capitalize()} ({solver.model_name}) …")
                t5 = time.time()
                try:
                    solution = solver.solve(prompt, system_prompt=system_prompt)
                    t5_elapsed = time.time() - t5
                    _log.info(f"       {len(solution)} char ({t5_elapsed:.1f}s)")

                    logger.log_layer5_attempt(solver_name, solver.model_name,
                                              len(solution), t5_elapsed, "ok")

                    if solution.strip():
                        if not best_solution:
                            best_solution = solution
                            best_provider = solver_name
                        if not self.ensemble:
                            break

                except Exception as e:
                    t5_elapsed = time.time() - t5
                    err_str = str(e)
                    _log.info(f"       [FAIL] {err_str[:80]} ({t5_elapsed:.1f}s)")
                    logger.log_layer5_attempt(solver_name, solver.model_name,
                                              0, t5_elapsed, "error", err_str[:200])
                    if any(k in err_str.lower() for k in ["credit", "quota", "insufficient", "billing"]):
                        _log.info(f"       [WARN] {solver_name} skipped for this PDF (quota)")
                        disabled_providers.add(solver_name)

            final_status = "error"
            if best_solution and best_provider:
                logger.log_layer5_best(best_provider, "ok")
                t6_final = time.time()
                extracted = self.layer6._extract_final_answer(best_solution)
                t6_elapsed = time.time() - t6_final
                logger.log_answer_extraction(extracted, t6_elapsed)

                result["layers"]["L5"] = {
                    "status": "OK", "provider": best_provider, "source": source,
                }
                disp = "OK" if extracted else "?"
                result["layers"]["L6"] = {
                    "final_answer": extracted,
                    "display_status": disp,
                }
                tag = "[OK]" if extracted else "[?]"
                _log.info(f"  [L6] {tag} extracted final line for display ({t6_elapsed:.3f}s)")
                final_status = "ok"
            else:
                result["layers"]["L5"] = {"status": "SKIP"}
                result["layers"]["L6"] = {"status": "skip", "display_status": "SKIP"}

            logger.finish_pdf(final_status)
            pipeline_results.append(result)

        stats = self._compute_stats(pipeline_results)
        self._print_pipeline_summary(pipeline_results, stats)

        log_path = logger.save()
        logger.print_summary()
        _log.info(f"\n  Log JSON: {log_path}")

        return logger.run_log

    @staticmethod
    def _compute_stats(pipeline_results: list[dict]) -> dict:
        l2_ok = sum(1 for r in pipeline_results if r["layers"]["L2"]["status"] == "OK")
        l3_ok = sum(1 for r in pipeline_results if r["layers"].get("L3", {}).get("status") == "OK")
        l5_ok = sum(1 for r in pipeline_results if r["layers"].get("L5", {}).get("status") == "OK")
        l6_extract_ok = sum(
            1 for r in pipeline_results
            if r["layers"].get("L6", {}).get("display_status") == "OK"
        )
        return {
            "nougat_ok": l2_ok,
            "vlm_ok": l3_ok,
            "llm_ok": l5_ok,
            "extract_ok": l6_extract_ok,
        }

    def _print_pipeline_summary(self, pipeline_results: list[dict], stats: dict):
        """Stdout table mirroring per-PDF layer status (complements the JSON log)."""
        _log.info(f"\n\n{'='*58}")
        _log.info("  BATCH RESULTS")
        _log.info(f"{'='*58}")

        l2_ok = stats["nougat_ok"]
        l5_ok = stats["llm_ok"]
        l6_ok = stats["extract_ok"]

        header = f"  {'File':<12} {'Nougat':<8} {'VLM':<8} {'Source':<16} {'LLM':<8} {'Answer':<8}"
        _log.info(f"\n{header}")
        _log.info(f"  {'─'*62}")
        for r in pipeline_results:
            f = r["file"]
            l2 = r["layers"]["L2"]["status"]
            l3 = r["layers"].get("L3", {}).get("status", "SKIP")
            l4_src = r["layers"].get("L4", {}).get("source", "-")
            l5 = r["layers"].get("L5", {}).get("status", "SKIP")
            l6 = r["layers"].get("L6", {}).get("display_status", "SKIP")
            _log.info(f"  {f:<12} {l2:<8} {l3:<8} {l4_src:<16} {l5:<8} {l6:<8}")

        total = len(pipeline_results)
        l3_ok = stats.get("vlm_ok", 0)
        _log.info(f"\n  Nougat OK:       {l2_ok}/{total}")
        _log.info(f"  VLM OK:          {l3_ok}/{total}")
        _log.info(f"  LLM OK:          {l5_ok}/{total}")
        _log.info(f"  Answer extracted:{l6_ok}/{total} (non-empty parsed final line)")
