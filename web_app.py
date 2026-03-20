"""
STEP Pipeline — Web UI
Flask backend with SSE streaming for real-time pipeline output.

Run:
    python web_app.py
    Open http://127.0.0.1:5000
"""

import threading
import queue
import sys
import time
import json
import re
from pathlib import Path

from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# Windows consoles often default to cp1252; force UTF-8 so box-drawing / checkmarks don't crash.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

_tasks: dict = {}
_thread_local = threading.local()
_original_stdout = sys.stdout
_solve_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Thread-safe stdout capture → SSE queue
# ---------------------------------------------------------------------------
class _SmartStdout:
    """Captures solver thread's stdout to a per-thread queue while
    preserving normal output on other threads."""

    def write(self, text):
        try:
            _original_stdout.write(text)
        except UnicodeEncodeError:
            enc = getattr(_original_stdout, "encoding", None) or "utf-8"
            try:
                _original_stdout.buffer.write(text.encode(enc, errors="replace"))
            except Exception:
                _original_stdout.write(text.encode("ascii", errors="replace").decode("ascii"))
        q = getattr(_thread_local, "queue", None)
        if q is not None:
            buf = getattr(_thread_local, "buf", "")
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                s = line.strip()
                if s:
                    q.put(_classify(s))
            _thread_local.buf = buf

    def flush(self):
        _original_stdout.flush()
        q = getattr(_thread_local, "queue", None)
        if q:
            buf = getattr(_thread_local, "buf", "")
            if buf.strip():
                q.put(_classify(buf.strip()))
            _thread_local.buf = ""


sys.stdout = _SmartStdout()


@app.after_request
def _no_cache_for_ui(response):
    """Avoid stale HTML/JS when the template was updated (browser cache)."""
    response.headers.setdefault("Cache-Control", "no-store, max-age=0, must-revalidate, private")
    response.headers.setdefault("Pragma", "no-cache")
    response.headers.setdefault("Expires", "0")
    return response


def _classify(line: str) -> dict:
    """Classify a log line into a typed SSE event."""
    m = re.match(r"\[L(\d+b?)\]\s*(.*)", line)
    if m:
        return {"type": "layer", "layer": m.group(1), "detail": m.group(2)}
    if "LLM:" in line or "VLM" in line:
        return {"type": "config", "text": line}
    if "Consensus" in line or "[OK]" in line:
        return {"type": "success", "text": line}
    if (
        "ERR" in line
        or "HATA" in line
        or "[FAIL]" in line
        or line.strip().startswith("FAIL ")
    ):
        return {"type": "error", "text": line}
    if "[WARN]" in line or "[RETRY]" in line:
        return {"type": "warning", "text": line}
    return {"type": "log", "text": line}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf")
    if not f or not f.filename:
        return jsonify(error="PDF file required"), 400

    safe = f.filename.replace("/", "_").replace("\\", "_")
    fp = UPLOAD_DIR / safe
    f.save(fp)

    tid = f"t{int(time.time() * 1000)}"
    q = queue.Queue()
    _tasks[tid] = {"queue": q, "done": False, "result": None}

    threading.Thread(target=_worker, args=(tid, fp), daemon=True).start()
    return jsonify(task_id=tid, filename=safe)


@app.route("/stream/<tid>")
def stream(tid):
    task = _tasks.get(tid)
    if not task:
        return jsonify(error="not found"), 404

    def gen():
        while True:
            try:
                msg = task["queue"].get(timeout=180)
                if msg is None:
                    break
                yield f"data: {json.dumps(msg, ensure_ascii=False, default=str)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/pdf/<path:name>")
def serve_pdf(name):
    return send_from_directory(UPLOAD_DIR, name)


# ---------------------------------------------------------------------------
# Background solver
# ---------------------------------------------------------------------------
def _worker(tid: str, filepath: Path):
    task = _tasks[tid]
    q = task["queue"]
    _thread_local.queue = q
    _thread_local.buf = ""

    with _solve_lock:
        try:
            from run import STEPSolver

            solver = STEPSolver(use_nougat=False, use_vlm=True)
            result = solver.solve(filepath, verbose=True)
            sys.stdout.flush()

            clean = {}
            for k, v in result.items():
                if isinstance(v, Path):
                    clean[k] = str(v)
                elif isinstance(v, list):
                    clean[k] = [str(x) for x in v]
                else:
                    clean[k] = v

            q.put({"type": "done", "data": clean})
            task["result"] = clean

        except Exception as e:
            sys.stdout.flush()
            q.put({"type": "error", "text": str(e)})

        finally:
            _thread_local.queue = None
            task["done"] = True
            q.put(None)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("  " + "=" * 44)
    print("  STEP Pipeline - Web UI")
    print("  http://127.0.0.1:5000")
    print("  " + "=" * 44)
    print()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
