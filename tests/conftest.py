from __future__ import annotations

from pathlib import Path

import pytest

from event_parser.config import load_config


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tmp_config(tmp_path):
    return load_config(
        config_path=str(REPO / "config" / "events.yaml"),
        storage_root=str(tmp_path / "storage"),
    )
