"""
JSON event log for batch runs: per-PDF layer timings and LLM attempts.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


class PipelineLogger:
    """Accumulates one structured log file per batch (`pipeline_log_*.json`)."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "pipeline_config": {},
            "environment": {},
            "models_used": [],
            "summary": {},
            "pdf_profiles": [],
        }
        self._current_pdf = None

    def log_config(self, config: dict):
        self.run_log["pipeline_config"] = config

    def log_environment(self, env: dict):
        self.run_log["environment"] = env

    def log_models(self, models: list[dict]):
        self.run_log["models_used"] = models

    # --- PDF profiling ---

    def start_pdf(self, filename: str):
        self._current_pdf = {
            "file_object": filename,
            "name": Path(filename).stem,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": None,
            "layers": {},
            "final_status": None,
        }
        self._pdf_start = time.time()

    def log_layer0(self, metadata: dict, text_pages: list, elapsed_s: float):
        self._current_pdf["file_metadata"] = {
            "file_size_bytes": int(metadata.get("file_size_kb", 0) * 1024),
            "pages_count": metadata.get("pages"),
            "author": metadata.get("author", ""),
            "producer": metadata.get("producer", ""),
            "creator": metadata.get("creator", ""),
            "creation_date": metadata.get("creation_date", ""),
            "modification_date": metadata.get("modification_date", ""),
        }
        total_chars = sum(len(p.get("text", "")) for p in text_pages)
        self._current_pdf["layers"]["L0_ingestion"] = {
            "technology": "PyMuPDF (fitz)",
            "status": "OK",
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "pages_extracted": len(text_pages),
                "total_chars": total_chars,
                "avg_chars_per_page": round(total_chars / len(text_pages)) if text_pages else 0,
            },
        }

    def log_layer1(self, profile: dict, elapsed_s: float):
        self._current_pdf["profiling"] = {
            "keywords": profile.get("keywords", []),
            "category": profile.get("category", "unknown"),
            "surface_type": profile.get("surface_type", "unknown"),
            "summary": profile.get("summary", ""),
        }
        self._current_pdf["file_metadata"].update({
            "file_size_bytes": profile.get("file_size_bytes", 0),
            "pages_count": profile.get("pages_count", 0),
            "author": profile.get("author", ""),
            "producer": profile.get("producer", ""),
            "creator": profile.get("creator", ""),
            "creation_date": profile.get("creation_date", ""),
            "modification_date": profile.get("modification_date", ""),
        })
        self._current_pdf["layers"]["L1_profiling"] = {
            "technology": "Heuristic + Regex Classifier",
            "status": "OK",
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "keywords_found": len(profile.get("keywords", [])),
                "category": profile.get("category", "unknown"),
                "surface_type": profile.get("surface_type", "unknown"),
            },
        }

    def log_layer2(self, nougat_result: dict, quality: dict, elapsed_s: float):
        char_count = nougat_result.get("char_count", 0)
        score = quality.get("score", 0)
        max_score = quality.get("max_score", 4)
        self._current_pdf["layers"]["L2_ocr"] = {
            "technology": "Nougat-OCR",
            "status": "OK" if score >= 2 else "FAIL",
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "latex_chars": char_count,
                "quality_score": f"{score}/{max_score}",
                "quality_checks": quality.get("checks", {}),
                "pages_processed": nougat_result.get("pages", 0),
                "output_path": nougat_result.get("output_path", ""),
            },
        }

    def log_layer3(self, vlm_result: dict, quality: dict, elapsed_s: float):
        char_count = vlm_result.get("char_count", 0)
        score = quality.get("score", 0)
        max_score = quality.get("max_score", 4)
        self._current_pdf["layers"]["L3_vlm"] = {
            "technology": "LLaMA 4 Scout 17B (Groq Vision)",
            "status": "OK" if score >= 2 else "FAIL",
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "vlm_chars": char_count,
                "quality_score": f"{score}/{max_score}",
                "quality_checks": quality.get("checks", {}),
                "pages_processed": vlm_result.get("pages", 0),
            },
        }

    def log_layer4(self, synthesis: dict, elapsed_s: float):
        self._current_pdf["layers"]["L4_synthesis"] = {
            "technology": "Multi-Source Fusion",
            "status": "OK",
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "source_strategy": synthesis.get("source", "unknown"),
                "nougat_score": synthesis.get("nougat_score", 0),
                "vlm_score": synthesis.get("vlm_score", 0),
                "prompt_chars": synthesis.get("prompt_chars", 0),
            },
        }

    def log_layer5_attempt(self, provider: str, model: str, solution_chars: int,
                           elapsed_s: float, status: str, error: str = None):
        layer5 = self._current_pdf["layers"].setdefault("L5_llm_solver", {
            "technology": "LLM Ensemble",
            "attempts": [],
            "best_provider": None,
            "best_status": None,
        })
        attempt = {
            "provider": provider,
            "model": model,
            "status": status,
            "elapsed_s": round(elapsed_s, 3),
            "solution_chars": solution_chars,
        }
        if error:
            attempt["error"] = error[:200]
        layer5["attempts"].append(attempt)

    def log_layer5_best(self, provider: str, status: str):
        layer5 = self._current_pdf["layers"].get("L5_llm_solver", {})
        layer5["best_provider"] = provider
        layer5["best_status"] = status

    def log_layer6(self, verification: dict, elapsed_s: float):
        self._current_pdf["layers"]["L6_verification"] = {
            "technology": "SymPy",
            "status": verification.get("status", "unknown"),
            "elapsed_s": round(elapsed_s, 3),
            "metrics": {
                "known_answer": str(verification.get("known", "")),
                "llm_answer": verification.get("llm_answer", ""),
                "method": verification.get("method", ""),
                "error_margin": verification.get("error"),
            },
        }

    def finish_pdf(self, final_status: str):
        self._current_pdf["elapsed_s"] = round(time.time() - self._pdf_start, 1)
        self._current_pdf["final_status"] = final_status
        self.run_log["pdf_profiles"].append(self._current_pdf)
        self._current_pdf = None

    # --- Summary ---

    def compute_summary(self):
        profiles = self.run_log["pdf_profiles"]
        total = len(profiles)

        l1_ok = sum(1 for p in profiles if p["layers"].get("L1_profiling", {}).get("status") == "OK")
        l2_ok = sum(1 for p in profiles if p["layers"].get("L2_ocr", {}).get("status") == "OK")
        l3_ok = sum(1 for p in profiles if p["layers"].get("L3_vlm", {}).get("status") == "OK")
        l4_ok = sum(1 for p in profiles if p["layers"].get("L4_synthesis", {}).get("status") == "OK")
        l5_ok = sum(1 for p in profiles if p["layers"].get("L5_llm_solver", {}).get("best_status") == "match")
        l6_match = sum(1 for p in profiles if p.get("final_status") == "match")
        l6_tested = sum(1 for p in profiles if p.get("final_status") in ("match", "mismatch"))

        total_time = sum(p.get("elapsed_s", 0) for p in profiles)
        nougat_time = sum(p["layers"].get("L2_ocr", {}).get("elapsed_s", 0) for p in profiles)
        vlm_time = sum(p["layers"].get("L3_vlm", {}).get("elapsed_s", 0) for p in profiles)

        # Per-model stats
        model_stats = {}
        for p in profiles:
            for attempt in p["layers"].get("L5_llm_solver", {}).get("attempts", []):
                prov = attempt["provider"]
                if prov not in model_stats:
                    model_stats[prov] = {"total": 0, "match": 0, "mismatch": 0, "error": 0,
                                         "total_time_s": 0, "total_chars": 0}
                model_stats[prov]["total"] += 1
                model_stats[prov]["total_time_s"] += attempt.get("elapsed_s", 0)
                model_stats[prov]["total_chars"] += attempt.get("solution_chars", 0)
                if attempt["status"] == "match":
                    model_stats[prov]["match"] += 1
                elif attempt["status"] == "mismatch":
                    model_stats[prov]["mismatch"] += 1
                elif "error" in attempt.get("status", ""):
                    model_stats[prov]["error"] += 1

        for prov, stats in model_stats.items():
            tested = stats["match"] + stats["mismatch"]
            stats["accuracy_pct"] = round(stats["match"] / tested * 100, 1) if tested else 0
            stats["avg_time_s"] = round(stats["total_time_s"] / stats["total"], 2) if stats["total"] else 0
            stats["total_time_s"] = round(stats["total_time_s"], 1)

        self.run_log["summary"] = {
            "total_pdfs": total,
            "total_elapsed_s": round(total_time, 1),
            "avg_per_pdf_s": round(total_time / total, 1) if total else 0,
            "layer_performance": {
                "L0_ingestion": {"technology": "PyMuPDF", "success_rate": f"{total}/{total}"},
                "L1_profiling": {"technology": "Heuristic Classifier", "success_rate": f"{l1_ok}/{total}"},
                "L2_ocr": {"technology": "Nougat-OCR", "success_rate": f"{l2_ok}/{total}",
                           "total_time_s": round(nougat_time, 1)},
                "L3_vlm": {"technology": "LLaMA 4 Scout (Groq Vision)", "success_rate": f"{l3_ok}/{total}",
                           "total_time_s": round(vlm_time, 1)},
                "L4_synthesis": {"technology": "Multi-Source Fusion", "success_rate": f"{l4_ok}/{total}"},
                "L5_llm_solver": {"technology": "LLM Ensemble", "success_rate": f"{l5_ok}/{total}"},
                "L6_verification": {"technology": "SymPy", "verified": f"{l6_match}/{l6_tested}",
                                     "accuracy_pct": round(l6_match / l6_tested * 100, 1) if l6_tested else 0},
            },
            "model_comparison": model_stats,
            "final_accuracy": {
                "match": l6_match,
                "total_testable": l6_tested,
                "accuracy_pct": round(l6_match / l6_tested * 100, 1) if l6_tested else 0,
            },
        }
        self.run_log["finished_at"] = datetime.now(timezone.utc).isoformat()

    def save(self) -> Path:
        self.compute_summary()
        log_path = self.output_dir / f"pipeline_log_{self.run_id}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.run_log, f, indent=2, default=str, ensure_ascii=False)
        return log_path

    def print_summary(self):
        s = self.run_log["summary"]
        print(f"\n{'='*66}")
        print(f"  PIPELINE LOG SUMMARY  (run_id: {self.run_id})")
        print(f"{'='*66}")
        print(f"\n  PDFs: {s['total_pdfs']}")
        print(f"  Wall time: {s['total_elapsed_s']}s ({s['total_elapsed_s']/60:.1f} min)")
        print(f"  Avg / PDF: {s['avg_per_pdf_s']}s")

        print(f"\n  --- Layer stats ---")
        for layer_name, layer_info in s["layer_performance"].items():
            tech = layer_info.get("technology", "")
            rate = layer_info.get("success_rate", "")
            extra = ""
            if "total_time_s" in layer_info:
                extra = f" | {layer_info['total_time_s']}s"
            if "accuracy_pct" in layer_info:
                extra += f" | %{layer_info['accuracy_pct']}"
            print(f"    {layer_name:<20} {tech:<15} {rate}{extra}")

        print(f"\n  --- LLM attempts ---")
        print(f"    {'Provider':<25} {'Match':>10} {'Avg s':>10} {'Total s':>10}")
        print(f"    {'─'*55}")
        for prov, stats in s.get("model_comparison", {}).items():
            tested = stats["match"] + stats["mismatch"]
            acc = f"{stats['match']}/{tested}" if tested else "N/A"
            print(f"    {prov:<25} {acc:>10} {stats['avg_time_s']:>9}s {stats['total_time_s']:>9}s")

        fa = s["final_accuracy"]
        print(f"\n  ═══ Verified: {fa['match']}/{fa['total_testable']} = %{fa['accuracy_pct']} ═══")
