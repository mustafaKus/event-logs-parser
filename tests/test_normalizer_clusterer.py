from event_parser.clusterer import cluster_signature
from event_parser.normalizer import normalize


def test_normalize_strips_ansi_and_whitespace():
    raw = "\x1b[31m2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456\x1b[0m\n"
    assert normalize(raw) == "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456"


def test_normalize_handles_non_string():
    assert normalize(None) == ""  # type: ignore[arg-type]


def test_clusterer_collapses_timestamp_and_ids():
    a = cluster_signature("2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    b = cluster_signature("2025-03-22T22:01:09Z INFO action=click user=u999 product=prod_001")
    assert a == b


def test_clusterer_distinguishes_different_events():
    a = cluster_signature("2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    b = cluster_signature("2025-01-15T10:23:50Z INFO action=add_to_cart user=u123 product=prod_456 qty=2")
    assert a != b


def test_clusterer_collapses_json_ids_and_numbers():
    a = cluster_signature('2025-01-15 11:00:00 app[web.1] {"event":"product_view","uid":"user-789","pid":"sku-001"}')
    b = cluster_signature('2025-01-15 11:00:05 app[web.1] {"event":"product_view","uid":"user-900","pid":"sku-999"}')
    assert a == b
