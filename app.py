from __future__ import annotations

import json as _json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from event_parser.config import load_config
from event_parser.pipeline import BatchPipeline
from event_parser.storage import ClusterIndex, LLMUsageStore, OutcomeWriter, ParserRepository


REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_LOGS_DIR = REPO_ROOT / "sample_logs"
UI_DIR = REPO_ROOT / "ui"


def create_app(config_path: str | None = None, storage_root: str | None = None) -> Flask:
    load_dotenv(REPO_ROOT / ".env", override=False)

    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )
    app = Flask(__name__)
    config = load_config(config_path=config_path, storage_root=storage_root)
    app.config["event_parser_config"] = config

    repo = ParserRepository(config.storage_root)
    clusters = ClusterIndex(config.storage_root)
    writer = OutcomeWriter(config.storage_root)
    llm_usage = LLMUsageStore(config.storage_root)

    # --- UI ---------------------------------------------------------------

    @app.get("/")
    def index():
        if not (UI_DIR / "index.html").exists():
            return jsonify({"error": "ui/index.html not found"}), 404
        return send_from_directory(UI_DIR, "index.html")

    @app.get("/ui/<path:filename>")
    def ui_static(filename: str):
        return send_from_directory(UI_DIR, filename)

    # --- API --------------------------------------------------------------

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "events": config.event_names})

    @app.get("/samples")
    def samples():
        """List sample logs shipped with the demo. Powers the UI showcase."""
        if not SAMPLE_LOGS_DIR.exists():
            return jsonify([])
        items = []
        for p in sorted(SAMPLE_LOGS_DIR.glob("*.log")):
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            items.append(
                {
                    "name": p.stem,
                    "path": str(p),
                    "line_count": sum(1 for line in lines if line.strip()),
                    "preview": [line for line in lines[:3]],
                }
            )
        return jsonify(items)

    @app.get("/datasets")
    def datasets():
        """List train/test paired datasets — one entry per (family, variant).

        Files follow the convention `{family}{N}_{train|test}.log`. Used by the
        UI to render the train→test learning showcase: each card runs the train
        file first, then the test file with the same customer_id, and surfaces
        parser hit-rate vs LLM call-rate on the test phase only.
        """
        if not SAMPLE_LOGS_DIR.exists():
            return jsonify([])
        # Pair up files by stem: {family}{N}_{split}
        import re as _re
        pairs: dict[tuple[str, int], dict] = {}
        pat = _re.compile(r"^([a-zA-Z]+)(\d+)_(train|test)$")
        for p in sorted(SAMPLE_LOGS_DIR.glob("*.log")):
            m = pat.match(p.stem)
            if not m:
                continue
            family, variant_s, split = m.group(1), int(m.group(2)), m.group(3)
            all_lines = [line for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            entry = pairs.setdefault((family, variant_s), {
                "family": family,
                "variant": variant_s,
                "customer_id": f"{family}{variant_s}",
                "train": None,
                "test": None,
            })
            entry[split] = {
                "name": p.stem,
                "path": str(p),
                "line_count": len(all_lines),
                # First line for the chip teaser. The modal "full view" fetches
                # the entire file via /sample/<name>.
                "preview": all_lines[:1],
            }
        # Family display order matches the storyboard.
        order = {"acme": 0, "globex": 1, "chaos": 2, "adversarial": 3}
        out = sorted(
            (e for e in pairs.values() if e["train"] and e["test"]),
            key=lambda e: (order.get(e["family"], 99), e["variant"]),
        )
        return jsonify(out)

    @app.get("/sample/<name>")
    def sample_content(name: str):
        path = SAMPLE_LOGS_DIR / f"{name}.log"
        if not path.exists():
            return jsonify({"error": "not found"}), 404
        return jsonify({"name": name, "content": path.read_text(encoding="utf-8", errors="replace")})

    @app.post("/process")
    def process():
        """Batch-process a log file.

        Accepts either:
          * multipart/form-data with a `file` field and `customer_id` form field
          * application/json: {"customer_id": "...", "path": "/abs/path/to/log.txt"}
          * application/json: {"customer_id": "...", "lines": ["line1", "line2"]}
          * application/json: {"customer_id": "...", "sample": "acme"}   # demo helper
        """
        pipeline = BatchPipeline(config)

        customer_id = None
        lines = None
        path = None

        if request.files.get("file"):
            customer_id = request.form.get("customer_id")
            if not customer_id:
                return jsonify({"error": "customer_id required"}), 400
            raw = request.files["file"].read().decode("utf-8", errors="replace")
            lines = raw.splitlines()
        else:
            body = request.get_json(silent=True) or {}
            customer_id = body.get("customer_id")
            if not customer_id:
                return jsonify({"error": "customer_id required"}), 400
            lines = body.get("lines")
            path = body.get("path")
            sample = body.get("sample")
            if sample and not path and lines is None:
                path = str(SAMPLE_LOGS_DIR / f"{sample}.log")

        if path:
            p = Path(path)
            if not p.exists():
                return jsonify({"error": f"path not found: {path}"}), 404
            stats = pipeline.process_file(p, customer_id)
        elif lines is not None:
            stats = pipeline.process_lines(lines, customer_id)
        else:
            return jsonify({"error": "provide file, lines, path, or sample"}), 400

        if stats.from_llm > 0:
            llm_usage.append(customer_id, {
                "calls": stats.from_llm,
                "input_tokens": stats.llm_input_tokens,
                "output_tokens": stats.llm_output_tokens,
            })

        result = stats.to_dict()
        result["llm_usage_totals"] = llm_usage.totals(customer_id)
        return jsonify(result)

    @app.post("/process/stream")
    def process_stream():
        """Same as /process but streams one SSE event per line as it's processed.
        Final event has type='done' with the batch summary."""
        pipeline = BatchPipeline(config)
        customer_id = None
        lines = None

        if request.files.get("file"):
            customer_id = request.form.get("customer_id")
            if not customer_id:
                return jsonify({"error": "customer_id required"}), 400
            raw = request.files["file"].read().decode("utf-8", errors="replace")
            lines = raw.splitlines()
        else:
            body = request.get_json(silent=True) or {}
            customer_id = body.get("customer_id")
            if not customer_id:
                return jsonify({"error": "customer_id required"}), 400
            lines = body.get("lines")
            path = body.get("path")
            sample = body.get("sample")
            if sample and not path and lines is None:
                path = str(SAMPLE_LOGS_DIR / f"{sample}.log")
            if path:
                p = Path(path)
                if not p.exists():
                    return jsonify({"error": f"path not found: {path}"}), 404
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines is None:
            return jsonify({"error": "provide file, lines, path, or sample"}), 400

        _cid = customer_id
        _lines = lines

        def generate():
            from event_parser.pipeline import BatchStats
            for item in pipeline.stream_lines(_lines, _cid):
                if isinstance(item, BatchStats):
                    if item.from_llm > 0:
                        llm_usage.append(_cid, {
                            "calls": item.from_llm,
                            "input_tokens": item.llm_input_tokens,
                            "output_tokens": item.llm_output_tokens,
                        })
                    done = item.to_dict()
                    done["type"] = "done"
                    done["llm_usage_totals"] = llm_usage.totals(_cid)
                    yield f"data: {_json.dumps(done)}\n\n"
                else:
                    yield f"data: {_json.dumps(item)}\n\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/reset/<customer_id>")
    def reset_customer(customer_id: str):
        """Clear all per-customer state. Intended for demo/showcase use."""
        removed = []
        for sub in ("parsers", "events", "quarantine", "clusters", "llm_usage"):
            target = config.storage_root / sub / f"{customer_id}.jsonl"
            if target.exists():
                target.unlink()
                removed.append(str(target.relative_to(config.storage_root)))
        return jsonify({"customer_id": customer_id, "removed": removed})

    @app.get("/parsers/<customer_id>")
    def list_parsers(customer_id: str):
        parsers = repo.list(customer_id)
        return jsonify([p.model_dump() for p in parsers])

    @app.get("/events/<customer_id>")
    def list_events(customer_id: str):
        limit = int(request.args.get("limit", 100))
        rows = writer.read_events(customer_id)
        return jsonify(rows[-limit:])

    @app.get("/quarantine/<customer_id>")
    def list_quarantine(customer_id: str):
        limit = int(request.args.get("limit", 100))
        rows = writer.read_quarantine(customer_id)
        return jsonify(rows[-limit:])

    @app.get("/clusters/<customer_id>")
    def list_clusters(customer_id: str):
        return jsonify(list(clusters.load(customer_id).values()))

    @app.get("/stats/<customer_id>")
    def stats(customer_id: str):
        events = writer.read_events(customer_id)
        quarantine = writer.read_quarantine(customer_id)
        parsers = repo.list(customer_id)
        by_status: dict[str, int] = {}
        for p in parsers:
            by_status[p.status] = by_status.get(p.status, 0) + 1
        by_source: dict[str, int] = {}
        for e in events:
            by_source[e.get("source", "?")] = by_source.get(e.get("source", "?"), 0) + 1
        total = len(events) + len(quarantine)
        return jsonify(
            {
                "customer_id": customer_id,
                "events_total": len(events),
                "quarantine_total": len(quarantine),
                "parsers_total": len(parsers),
                "parsers_by_status": by_status,
                "events_by_source": by_source,
                "llm_call_rate": (by_source.get("llm", 0) / total) if total else 0.0,
                "parser_hit_rate": (by_source.get("parser", 0) / total) if total else 0.0,
                "quarantine_rate": (len(quarantine) / total) if total else 0.0,
                "llm_usage_totals": llm_usage.totals(customer_id),
            }
        )

    return app


if __name__ == "__main__":
    load_dotenv(REPO_ROOT / ".env", override=False)
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    create_app().run(host=host, port=port, debug=debug)
