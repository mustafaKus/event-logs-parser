# Event Logs Parser

`event-logs-parser` is a small Flask app and Python pipeline for turning messy product-analytics log lines into canonical events.

It is designed to:

- ingest tenant-specific log files,
- extract known event types like `click`, `view`, and `purchase`,
- learn reusable parsers from previously seen log shapes,
- reduce LLM calls over time by routing repeat patterns to generated parsers,
- quarantine lines that do not match the configured event contract.

By default, the repo runs fully offline with a deterministic mock extractor. You can optionally switch to an OpenAI-backed extractor through environment variables.

## What it does

For each customer dataset, the pipeline:

1. normalizes each raw line,
2. clusters similar log shapes,
3. tries an existing learned parser first,
4. falls back to an extractor when needed,
5. stores extracted events, quarantined rows, parser definitions, cluster metadata, and LLM usage.

The canonical event schema lives in `config/events.yaml`.

## Supported event types

Out of the box, the repo recognizes these canonical events:

- `click`
- `add_to_cart`
- `remove_from_cart`
- `view`
- `purchase`

Each event has required and optional fields configured in `config/events.yaml`.

## Repo highlights

- `app.py` — Flask app, API routes, and demo UI wiring
- `event_parser/pipeline.py` — batch pipeline and stats
- `event_parser/router.py` — parser vs extractor routing logic
- `event_parser/agents/extractor.py` — offline mock extractor and optional OpenAI extractor
- `event_parser/storage.py` — JSONL-backed persistence for parsers, events, quarantine, clusters, and usage
- `scripts/demo.py` — cold vs warm run showcase
- `sample_logs/` — demo and benchmark log fixtures
- `ui/index.html` — browser demo UI
- `tests/` — pytest coverage for the pipeline, API, generators, and adversarial cases

## Quick start

### 1) Install dependencies

```zsh
make install
```

### 2) Create a local environment file

```zsh
make env
```

That copies `.env.example` to `.env` if needed.

### 3) Run the app

```zsh
make up
```

By default the app starts on:

- UI: `http://127.0.0.1:5001`
- Health check: `http://127.0.0.1:5001/health`

## Default runtime mode

The project defaults to the offline-safe mock extractor unless `EVENT_PARSER_LLM=openai` is set.

This means you can run the app, tests, and demo without an API key.

## Configuration

The main environment variables are:

- `EVENT_PARSER_CONFIG` — path to the YAML event catalog, default `config/events.yaml`
- `EVENT_PARSER_STORAGE` — storage root for JSONL data, default `storage`
- `EVENT_PARSER_LLM` — extractor backend: `mock` or `openai`
- `EVENT_PARSER_MODEL` — model identifier for the OpenAI extractor path
- `OPENAI_API_KEY` — only required when using `EVENT_PARSER_LLM=openai`
- `FLASK_HOST`, `FLASK_PORT`, `FLASK_DEBUG` — Flask runtime settings

### Event catalog

`config/events.yaml` defines:

- canonical event names,
- required and optional fields per event,
- lifecycle thresholds for parser promotion,
- maximum line size guardrails.

## Using the pipeline

### Run the demo showcase

```zsh
make demo
```

This runs `scripts/demo.py`, processes `sample_logs/acme.log`, and shows the difference between a cold pass and a warm pass after parsers have been learned.

### Generate synthetic logs

```zsh
make logs ROWS=5000 OUT=sample_logs/chaos.log SEED=42
```

### Regenerate train/test sample datasets

```zsh
make samples
```

## HTTP API

### Core endpoints

- `GET /` — demo UI
- `GET /health` — service health and configured event names
- `GET /samples` — list shipped sample log files
- `GET /datasets` — list paired `*_train.log` / `*_test.log` showcase datasets
- `GET /sample/<name>` — return the contents of a sample log
- `POST /process` — batch process lines, a file upload, or a file path
- `POST /process/stream` — stream per-line results via server-sent events

### Inspection endpoints

- `GET /parsers/<customer_id>` — learned parsers for a tenant
- `GET /events/<customer_id>` — extracted event rows
- `GET /quarantine/<customer_id>` — quarantined rows
- `GET /clusters/<customer_id>` — observed cluster signatures
- `GET /stats/<customer_id>` — tenant summary metrics
- `POST /reset/<customer_id>` — clear stored state for a tenant

### Example: process inline lines

```zsh
curl -X POST http://127.0.0.1:5001/process \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "acme",
    "lines": [
      "2025-01-15T10:23:45Z INFO action=click user=u123 product=prod_456",
      "2025-01-15T10:23:46Z INFO action=click user=u124 product=prod_457"
    ]
  }'
```

### Example: process a shipped sample

```zsh
curl -X POST http://127.0.0.1:5001/process \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"acme","sample":"acme"}'
```

### Example: inspect stats

```zsh
curl http://127.0.0.1:5001/stats/acme
```

## Storage model

Per-customer state is stored under `storage/` as JSONL files:

- `storage/parsers/` — learned parser library
- `storage/events/` — successful extractions
- `storage/quarantine/` — rejected or unknown lines
- `storage/clusters/` — cluster signatures and counts
- `storage/llm_usage/` — accumulated call and token metrics

This makes the project easy to inspect and reset during demos.

## Testing

Run the full test suite with:

```zsh
make test
```

The tests cover:

- Flask endpoints,
- parser learning and promotion,
- adversarial and malformed inputs,
- generators and normalizer behavior,
- tenant isolation and repeat-run improvements.

## Reset and cleanup

Clear stored runtime data only:

```zsh
make reset
```

Remove caches and generated storage:

```zsh
make clean
```

Remove caches, storage, and the virtual environment:

```zsh
make distclean
```

## OpenAI-backed mode

If you want to use the real extractor path instead of the mock one:

1. install the optional OpenAI Agents dependency,
2. set `EVENT_PARSER_LLM=openai`,
3. add `OPENAI_API_KEY` to `.env`.

The repository currently leaves the OpenAI package commented out in `requirements.txt`, so the offline path remains the default developer experience.

## Development notes

- The Flask app factory is `create_app()` in `app.py`.
- The default storage format is intentionally simple and local-first.
- The mock extractor is deterministic so tests and demos run reliably.
- Oversized lines are quarantined using the `max_line_bytes` guardrail.
- Event extraction is schema-constrained by the configured event catalog.

## Typical workflow

```zsh
make install
make env
make up
```

In a second terminal:

```zsh
curl -X POST http://127.0.0.1:5001/process \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"globex","sample":"globex"}'

curl http://127.0.0.1:5001/stats/globex
```