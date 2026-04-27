from pathlib import Path

from event_parser.agents import MockExtractor
from event_parser.pipeline import BatchPipeline


REPO = Path(__file__).resolve().parents[1]


def test_acme_batch_extracts_and_promotes_parsers(tmp_config):
    pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
    stats = pipeline.process_file(REPO / "sample_logs" / "acme.log", customer_id="acme")
    assert stats.total_lines > 0
    assert stats.events_extracted > 0
    # At least the click/view/add_to_cart/remove_from_cart clusters should be seen.
    assert stats.clusters_seen >= 4
    # First occurrence of each cluster goes to LLM; subsequent ones should be cheaper.
    assert stats.from_parser + stats.from_llm == stats.events_extracted
    # Adversarial DEBUG/WARN lines are not canonical events → quarantined.
    assert stats.quarantined >= 1

    # After the batch, at least one shadow parser should have been promoted to active.
    parsers = pipeline.repo.list("acme")
    assert parsers, "expected minted parsers"
    assert any(p.status == "active" for p in parsers), (
        "at least one shadow parser should promote to active within the batch"
    )


def test_globex_json_batch(tmp_config):
    pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
    stats = pipeline.process_file(REPO / "sample_logs" / "globex.log", customer_id="globex")
    assert stats.events_extracted > 0
    assert stats.quarantined >= 1  # page_view, heartbeat etc.
    parsers = pipeline.repo.list("globex")
    assert any(p.pattern_type == "jsonpath" for p in parsers)


def test_second_batch_avoids_most_llm_calls(tmp_config):
    pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
    pipeline.process_file(REPO / "sample_logs" / "acme.log", customer_id="acme")

    # Run the same file again — now parsers should dominate.
    stats = pipeline.process_file(REPO / "sample_logs" / "acme.log", customer_id="acme")
    assert stats.from_parser > stats.from_llm, (
        f"expected parser-dominated second run, got parser={stats.from_parser} llm={stats.from_llm}"
    )


def test_multi_tenant_parsers_do_not_leak(tmp_config):
    pipeline = BatchPipeline(tmp_config, extractor=MockExtractor())
    pipeline.process_file(REPO / "sample_logs" / "acme.log", customer_id="acme")
    pipeline.process_file(REPO / "sample_logs" / "globex.log", customer_id="globex")
    acme_parsers = pipeline.repo.list("acme")
    globex_parsers = pipeline.repo.list("globex")
    assert {p.customer_id for p in acme_parsers} == {"acme"}
    assert {p.customer_id for p in globex_parsers} == {"globex"}
    # Patterns should not overlap across tenants.
    assert not (
        {p.pattern_body for p in acme_parsers} & {p.pattern_body for p in globex_parsers}
    )


def test_idempotent_stats(tmp_config):
    pipeline_a = BatchPipeline(tmp_config, extractor=MockExtractor())
    stats_a = pipeline_a.process_file(REPO / "sample_logs" / "acme.log", customer_id="a1")

    # Fresh config + fresh storage, separate customer id → same extraction outcome.
    pipeline_b = BatchPipeline(tmp_config, extractor=MockExtractor())
    stats_b = pipeline_b.process_file(REPO / "sample_logs" / "acme.log", customer_id="a2")
    assert stats_a.total_lines == stats_b.total_lines
    assert stats_a.events_extracted == stats_b.events_extracted
    assert stats_a.quarantined == stats_b.quarantined
