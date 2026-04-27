from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EventSpec:
    name: str
    required_fields: list[str]
    optional_fields: list[str] = field(default_factory=list)

    @property
    def all_fields(self) -> list[str]:
        return [*self.required_fields, *self.optional_fields]


@dataclass
class LifecycleConfig:
    shadow_to_active_matches: int = 3
    shadow_to_active_agreement: float = 0.95
    max_line_bytes: int = 8192


@dataclass
class AppConfig:
    events: dict[str, EventSpec]
    lifecycle: LifecycleConfig
    storage_root: Path
    config_path: Path

    @property
    def event_names(self) -> list[str]:
        return sorted(self.events.keys())


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    storage_root: str | os.PathLike[str] | None = None,
) -> AppConfig:
    cfg_path = Path(config_path or os.environ.get("EVENT_PARSER_CONFIG", "config/events.yaml")).resolve()
    with cfg_path.open() as f:
        raw = yaml.safe_load(f)

    events = {
        name: EventSpec(
            name=name,
            required_fields=list(spec.get("required_fields", [])),
            optional_fields=list(spec.get("optional_fields", [])),
        )
        for name, spec in raw.get("events", {}).items()
    }
    lifecycle_raw = raw.get("lifecycle", {})
    lifecycle = LifecycleConfig(
        shadow_to_active_matches=int(lifecycle_raw.get("shadow_to_active_matches", 3)),
        shadow_to_active_agreement=float(lifecycle_raw.get("shadow_to_active_agreement", 0.95)),
        max_line_bytes=int(lifecycle_raw.get("max_line_bytes", 8192)),
    )

    root = Path(storage_root or os.environ.get("EVENT_PARSER_STORAGE", "storage")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("parsers", "events", "quarantine", "clusters"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    return AppConfig(events=events, lifecycle=lifecycle, storage_root=root, config_path=cfg_path)
