"""Cold→warm batch showcase. Run via `make demo`."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from event_parser.agents import MockExtractor
from event_parser.config import load_config
from event_parser.pipeline import BatchPipeline


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="event-parser-demo-")
    try:
        cfg = load_config(
            config_path=str(REPO / "config" / "events.yaml"),
            storage_root=tmp,
        )
        for label in ("COLD", "WARM"):
            pipeline = BatchPipeline(cfg, extractor=MockExtractor())
            stats = pipeline.process_file(
                REPO / "sample_logs" / "acme.log",
                customer_id="acme",
            )
            print(f"\n=== {label} ===")
            print(json.dumps(stats.to_dict(), indent=2))
        print("\nParser library after warm pass:")
        for p in pipeline.repo.list("acme"):
            print(
                f"  {p.event_type:<20s} status={p.status:<8s} "
                f"type={p.pattern_type:<8s} observed={p.observed_count}"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
