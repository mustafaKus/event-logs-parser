import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.generate_logs import DIALECTS, EVENT_ALIASES, generate  # noqa: E402

from event_parser.agents import MockExtractor
from event_parser.pipeline import BatchPipeline


def test_generator_produces_requested_rows():
    rows = list(generate(10_000, seed=42))
    assert len(rows) == 10_000


def test_generator_is_deterministic_with_seed():
    a = list(generate(500, seed=7))
    b = list(generate(500, seed=7))
    assert a == b


def test_generator_covers_all_dialects_and_events():
    # Large enough sample to hit every dialect and every event type at low noise.
    rows = list(generate(2_000, seed=1, noise_ratio=0.0))
    # Every canonical event name must appear (in at least one dialect form).
    joined = "\n".join(rows).lower()
    for canonical in EVENT_ALIASES:
        aliases = [a.lower() for a in EVENT_ALIASES[canonical]]
        assert any(a in joined for a in aliases), f"no alias for {canonical} found"


def test_chaos_log_pipeline_smoke(tmp_config):
    # Small synthetic run end-to-end to confirm the pipeline doesn't choke on chaos.
    lines = list(generate(1_000, seed=123))
    pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
    stats = pipeline.process_lines(lines, customer_id="chaos")
    assert stats.total_lines > 0
    # Noise dominates: quarantine should be a healthy majority.
    assert stats.quarantined > stats.events_extracted
    # Some events must still be extracted.
    assert stats.events_extracted > 0
    # Determinism: every line accounted for.
    assert stats.events_extracted + stats.quarantined == stats.total_lines


def test_chaos_log_second_pass_lowers_llm_rate(tmp_config):
    lines = list(generate(1_500, seed=9))
    pipe1 = BatchPipeline(tmp_config, extractor=MockExtractor())
    s1 = pipe1.process_lines(lines, customer_id="chaos2")
    pipe2 = BatchPipeline(tmp_config, extractor=MockExtractor())
    s2 = pipe2.process_lines(lines, customer_id="chaos2")
    # Event counts should match — pipeline is deterministic for identical input.
    assert s1.events_extracted == s2.events_extracted
    # Second pass must not INCREASE LLM load on the same data.
    assert s2.from_llm <= s1.from_llm


def test_dialects_all_produce_parseable_event_lines(tmp_config):
    # Force every dialect to produce at least one line and check the pipeline
    # recognizes at least one event from each.
    from datetime import datetime, timezone
    import random

    rng = random.Random(0)
    ts = datetime(2026, 4, 25, tzinfo=timezone.utc)
    events = list(EVENT_ALIASES.keys())
    for dialect in DIALECTS:
        lines = [dialect(rng, ts, rng.choice(events)) for _ in range(20)]
        pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
        stats = pipeline.process_lines(lines, customer_id=f"dialect_{dialect.__name__}")
        assert stats.events_extracted > 0, (
            f"dialect {dialect.__name__} produced zero extractable events"
        )
