import io
from pathlib import Path

import pytest

from app import create_app


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        config_path=str(REPO / "config" / "events.yaml"),
        storage_root=str(tmp_path / "storage"),
    )
    app.testing = True
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "click" in body["events"]


def test_process_via_json_lines(client):
    resp = client.post(
        "/process",
        json={
            "customer_id": "acme",
            "lines": [
                "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456",
                "2025-01-15T10:23:46Z INFO action=click user=u124 product=prod_457",
                "2025-01-15T10:23:47Z INFO action=click user=u125 product=prod_458",
                "2025-01-15T10:23:48Z INFO action=click user=u126 product=prod_459",
            ],
        },
    )
    assert resp.status_code == 200
    stats = resp.get_json()
    assert stats["total_lines"] == 4
    assert stats["events_extracted"] == 4


def test_process_via_upload(client):
    data = {
        "customer_id": "acme",
        "file": (io.BytesIO(b"2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456\n"), "log.txt"),
    }
    resp = client.post("/process", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    stats = resp.get_json()
    assert stats["total_lines"] == 1
    assert stats["events_extracted"] == 1


def test_stats_endpoint(client):
    client.post(
        "/process",
        json={
            "customer_id": "globex",
            "lines": [
                '2025-01-15 11:00:00 app[web.1] {"event":"product_view","uid":"user-789","pid":"sku-001"}',
                '2025-01-15 11:00:02 app[web.1] heartbeat ok',
            ],
        },
    )
    resp = client.get("/stats/globex")
    body = resp.get_json()
    assert body["customer_id"] == "globex"
    assert body["events_total"] >= 1
    assert body["quarantine_total"] >= 1


def test_missing_customer_id_rejected(client):
    resp = client.post("/process", json={"lines": ["whatever"]})
    assert resp.status_code == 400
