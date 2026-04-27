"""Generate train/test split sample logs for each demo family.

Produces, in sample_logs/:
    {family}{N}_train.log    {family}{N}_test.log

For each family in {acme, globex, chaos, adversarial} and each variant
N in {1, 2}. Variants share the same dialect family but use a *different*
alias profile (different field-name spellings, different event-name spellings)
so that running variant 2 after variant 1 demonstrates whether learned
parsers transfer to a slightly shifted vocabulary.

Each dataset is built with a narrow alias set so cluster signatures collapse
to a small number of templates. That way the training pass actually crosses
the shadow→active threshold (`shadow_to_active_matches=3`,
`shadow_to_active_agreement=0.95`) and the test pass shows a high parser
hit-rate.

Usage:
    python scripts/generate_samples.py
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse noise + adversarial machinery from the original generator.
from generate_logs import (
    NOISE_TEMPLATES,
    _adversarial_noise,
    _fill_noise,
)


# -- per-dataset alias profiles ---------------------------------------------
# Two variants per family; same dialect, different alias spellings. Narrow
# (one or two choices each) so clustering collapses into a few templates.

CURRENCIES = ["USD", "EUR", "GBP"]


def _profile(
    user_keys, product_keys, order_keys, total_keys, qty_keys, events,
):
    return {
        "user_keys": user_keys,
        "product_keys": product_keys,
        "order_keys": order_keys,
        "total_keys": total_keys,
        "qty_keys": qty_keys,
        "events": events,
    }


# Variant 1 across all families: snake_case-ish spellings.
_PROFILE_V1 = _profile(
    user_keys=["user_id"],
    product_keys=["product_id"],
    order_keys=["order_id"],
    total_keys=["total"],
    qty_keys=["qty"],
    events={
        "click": ["click"],
        "view": ["view"],
        "add_to_cart": ["add_to_cart"],
        "remove_from_cart": ["remove_from_cart"],
        "purchase": ["purchase"],
    },
)

# Variant 2 across all families: camelCase / business-y spellings.
_PROFILE_V2 = _profile(
    user_keys=["customerId"],
    product_keys=["sku"],
    order_keys=["orderNumber"],
    total_keys=["amount"],
    qty_keys=["quantity"],
    events={
        "click": ["Click"],
        "view": ["ProductViewed"],
        "add_to_cart": ["AddToCart"],
        "remove_from_cart": ["RemoveFromCart"],
        "purchase": ["CheckoutComplete"],
    },
)


# -- dialect renderers (narrow / parameterised by profile) ------------------


def _rand_id(rng: random.Random, prefix: str = "") -> str:
    return f"{prefix}{rng.randint(1, 99999)}"


def _rand_hex(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choices("0123456789abcdef", k=n))


def _dialect_a(rng: random.Random, ts: datetime, event: str, prof: dict) -> str:
    """ACME-style: ISO ts + INFO + key=value pairs."""
    alias = rng.choice(prof["events"][event])
    uk = rng.choice(prof["user_keys"])
    pk = rng.choice(prof["product_keys"])
    parts = [
        ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "INFO",
        f"action={alias}",
        f"{uk}=u{rng.randint(1, 9999)}",
    ]
    if event == "purchase":
        parts.append(f"{rng.choice(prof['order_keys'])}=ord-{rng.randint(1, 99999)}")
        parts.append(f"{rng.choice(prof['total_keys'])}={rng.uniform(5, 500):.2f}")
        parts.append(f"currency={rng.choice(CURRENCIES)}")
    else:
        parts.append(f"{pk}=prod_{rng.randint(100, 9999)}")
    if event == "add_to_cart":
        parts.append(f"{rng.choice(prof['qty_keys'])}={rng.randint(1, 5)}")
    return " ".join(parts)


def _dialect_b(rng: random.Random, ts: datetime, event: str, prof: dict) -> str:
    """Globex-style: textual prefix + embedded JSON."""
    alias = rng.choice(prof["events"][event])
    uk = rng.choice(prof["user_keys"])
    pk = rng.choice(prof["product_keys"])
    payload: dict = {"event": alias, uk: f"user-{rng.randint(1, 9999)}"}
    if event == "purchase":
        payload[rng.choice(prof["order_keys"])] = f"ord-{rng.randint(1, 99999)}"
        payload[rng.choice(prof["total_keys"])] = f"{rng.uniform(5, 500):.2f}"
        payload["currency"] = rng.choice(CURRENCIES)
    else:
        payload[pk] = f"sku-{rng.randint(100, 9999)}"
    if event == "add_to_cart":
        payload[rng.choice(prof["qty_keys"])] = rng.randint(1, 5)
    prefix = ts.strftime("%Y-%m-%d %H:%M:%S") + f" app[web.{rng.randint(1, 5)}]"
    return f"{prefix} {json.dumps(payload)}"


def _dialect_c(rng: random.Random, ts: datetime, event: str, prof: dict) -> str:
    """Whole-line JSON with camelCase keys (used by chaos for variety)."""
    alias = rng.choice(prof["events"][event])
    uk = rng.choice(prof["user_keys"])
    payload: dict = {
        "timestamp": ts.isoformat(),
        "eventType": alias,
        uk: f"u{rng.randint(1, 9999)}",
    }
    if event == "purchase":
        payload[rng.choice(prof["order_keys"])] = f"ord-{rng.randint(1, 99999)}"
        payload[rng.choice(prof["total_keys"])] = round(rng.uniform(5, 500), 2)
        payload["currency"] = rng.choice(CURRENCIES)
    else:
        payload[rng.choice(prof["product_keys"])] = f"p{rng.randint(1, 9999)}"
    if event == "add_to_cart":
        payload[rng.choice(prof["qty_keys"])] = rng.randint(1, 5)
    return json.dumps(payload)


_DIALECT_FNS = {"a": _dialect_a, "b": _dialect_b, "c": _dialect_c}


# -- per-family dataset specs -----------------------------------------------

CANONICAL_EVENTS = ["click", "view", "add_to_cart", "remove_from_cart", "purchase"]


# Each family chooses which dialects + how much noise + adversarial pressure.
FAMILIES: dict[str, dict] = {
    "acme":        {"dialects": ["a"],            "noise": 0.15, "adversarial": 0.0,
                    "size": {"train": 240, "test": 120}},
    "globex":      {"dialects": ["b"],            "noise": 0.15, "adversarial": 0.0,
                    "size": {"train": 240, "test": 120}},
    "chaos":       {"dialects": ["a", "b", "c"],  "noise": 0.55, "adversarial": 0.02,
                    "size": {"train": 1500, "test": 700}},
    "adversarial": {"dialects": ["a"],            "noise": 0.10, "adversarial": 0.40,
                    "size": {"train": 240, "test": 120}},
}


# -- generator --------------------------------------------------------------


def _generate(spec: dict, profile: dict, rows: int, seed: int):
    rng = random.Random(seed)
    ts = datetime(2026, 4, 25, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=seed)
    dialects = spec["dialects"]
    for _ in range(rows):
        ts += timedelta(milliseconds=rng.randint(10, 8000))
        if rng.random() < spec["noise"]:
            line = _fill_noise(rng.choice(NOISE_TEMPLATES), rng)
        else:
            event = rng.choice(CANONICAL_EVENTS)
            dialect = _DIALECT_FNS[rng.choice(dialects)]
            line = dialect(rng, ts, event, profile)
        if rng.random() < spec["adversarial"]:
            line = _adversarial_noise(rng, line)
        yield line


# Stable per-(family, variant, split) seeds → reproducible files.
_FAMILY_BASE = {"acme": 1000, "globex": 2000, "chaos": 3000, "adversarial": 4000}
_SPLIT_OFFSET = {"train": 0, "test": 500}


def _seed_for(family: str, variant: int, split: str) -> int:
    return _FAMILY_BASE[family] + variant * 100 + _SPLIT_OFFSET[split]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "sample_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for family, spec in FAMILIES.items():
        for variant, profile in [(1, _PROFILE_V1), (2, _PROFILE_V2)]:
            for split in ("train", "test"):
                rows = spec["size"][split]
                seed = _seed_for(family, variant, split)
                path = out_dir / f"{family}{variant}_{split}.log"
                with path.open("w", encoding="utf-8") as f:
                    for line in _generate(spec, profile, rows, seed):
                        f.write(line + "\n")
                written.append((path, rows))
                print(f"wrote {rows:>5d} rows → {path.name}")
    print(f"\nDone. {len(written)} files written to {out_dir}.")


if __name__ == "__main__":
    main()
