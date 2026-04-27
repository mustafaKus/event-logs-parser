import json

import pytest

from event_parser.models import Parser
from event_parser.parsers import compile_grok, compile_regex, execute_parser, validate_parser


def test_compile_grok_word_and_timestamp():
    rx = compile_grok("%{TIMESTAMP:ts} INFO action=click user=%{WORD:user} product=%{NOTSPACE:product_id}")
    m = rx.search("2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    assert m is not None
    assert m.group("ts") == "2025-01-15T10:23:45Z"
    assert m.group("user") == "u123"
    assert m.group("product_id") == "prod_456"


def test_compile_grok_supports_timestamp_iso8601_alias():
    rx = compile_grok("%{TIMESTAMP_ISO8601:ts} INFO action=click user=%{WORD:user}")
    m = rx.search("2025-01-15T10:23:45Z INFO action=click user=u123")
    assert m is not None
    assert m.group("ts") == "2025-01-15T10:23:45Z"
    assert m.group("user") == "u123"


def test_compile_regex_supports_pcre_named_groups():
    rx = compile_regex(r"^(?<timestamp>\S+) INFO action=click user=(?<user_id>\S+) product=(?<product_id>\S+)$")
    m = rx.search("2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    assert m is not None
    assert m.group("timestamp") == "2025-01-15T10:23:45Z"
    assert m.group("user_id") == "u123"
    assert m.group("product_id") == "prod_456"


def test_execute_grok_parser():
    p = Parser(
        customer_id="t",
        event_type="click",
        pattern_type="grok",
        pattern_body="%{TIMESTAMP:timestamp} INFO action=click user=%{WORD:user_id} product=%{NOTSPACE:product_id}",
    )
    fields = execute_parser(p, "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    assert fields == {
        "timestamp": "2025-01-15T10:23:45Z",
        "user_id": "u123",
        "product_id": "prod_456",
    }


def test_execute_grok_parser_miss():
    p = Parser(
        customer_id="t",
        event_type="click",
        pattern_type="grok",
        pattern_body="%{TIMESTAMP:timestamp} INFO action=click user=%{WORD:user_id} product=%{NOTSPACE:product_id}",
    )
    assert execute_parser(p, "unrelated line") is None


def test_execute_kv_parser():
    body = json.dumps({"separator": " ", "kv_sep": "=", "fields": ["user", "product", "action"]})
    p = Parser(customer_id="t", event_type="click", pattern_type="kv", pattern_body=body)
    fields = execute_parser(p, "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456")
    assert fields["action"] == "click"
    assert fields["user"] == "u123"
    assert fields["product"] == "prod_456"


def test_execute_jsonpath_parser():
    body = json.dumps(
        {
            "fields": {
                "user_id": "$.uid",
                "product_id": "$.pid",
                "event": "$.event",
            }
        }
    )
    p = Parser(customer_id="t", event_type="view", pattern_type="jsonpath", pattern_body=body)
    fields = execute_parser(
        p,
        '2025-01-15 11:00:00 app[web.1] {"event":"product_view","uid":"user-789","pid":"sku-001"}',
    )
    assert fields == {"user_id": "user-789", "product_id": "sku-001", "event": "product_view"}


def test_validate_parser_requires_all_samples_match():
    p = Parser(
        customer_id="t",
        event_type="click",
        pattern_type="grok",
        pattern_body="%{TIMESTAMP:timestamp} INFO action=click user=%{WORD:user_id} product=%{NOTSPACE:product_id}",
    )
    samples = [
        "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456",
        "2025-01-15T10:23:46Z INFO action=click user=u124 product=prod_457",
    ]
    expected = {"user_id": "u123", "product_id": "prod_456"}
    assert validate_parser(p, samples, expected) is True


def test_validate_parser_rejects_mismatched_sample():
    p = Parser(
        customer_id="t",
        event_type="click",
        pattern_type="grok",
        pattern_body="%{TIMESTAMP:timestamp} INFO action=click user=%{WORD:user_id} product=%{NOTSPACE:product_id}",
    )
    samples = [
        "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456",
        "random unrelated line",
    ]
    expected = {"user_id": "u123", "product_id": "prod_456"}
    assert validate_parser(p, samples, expected) is False
