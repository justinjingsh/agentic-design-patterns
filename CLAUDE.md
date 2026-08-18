# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**Dependencies & Setup**
```bash
uv sync                    # Install/sync all dependencies
uv run app                 # Run main application
uv run python -m src.examples prompt_chaining  # Run prompt chaining example
uv run python -m src.examples help             # List all available examples
```

**Project is managed with `uv`** — a fast Python package manager. All Python commands should use `uv run` prefix.

## Architecture Overview

### Modular Example-Based Structure

This is a **learning/demo project** organized around design patterns, not a monolithic application:

```
src/
├── app/              # Core application (config, validation, entry point)
├── examples/         # All design pattern examples (loosely coupled)
│   ├── cli.py        # Shared CLI utilities (help text, example registry)
│   ├── __main__.py   # CLI dispatcher (routes commands to examples)
│   └── [pattern]/    # Each pattern gets its own module (prompt_chaining, routing, etc.)
```

### Key Design Decisions

1. **Examples are First-Class**: Each example is a standalone module under `src/examples/`. They don't import from each other; they only share utilities via `cli.py`.

2. **Shared CLI Registry**: `src/examples/cli.py` defines an `EXAMPLES` dictionary that centralizes:
   - Available examples and their descriptions
   - Help text generation (auto-updates as examples are added)
   - This is the single source of truth for CLI documentation

3. **Entry Point Pattern**: 
   - `app.py` (project root) → wrapper to find `src` module
   - `src.app:main()` → validates AWS credentials, shows usage
   - `src.examples:main()` → CLI dispatcher to run individual examples

4. **Lazy Imports**: Example code imports only when the command is invoked (keeps startup fast, avoids loading unused dependencies).

## Adding New Examples

**Minimal steps:**

1. Create a new module: `src/examples/routing/router.py`
2. Export your function: `src/examples/routing/__init__.py` → `from .router import run_routing`
3. Register in `src/examples/cli.py` → `EXAMPLES["routing"] = "description"`
4. Add handler in `src/examples/__main__.py` → `elif command == "routing": from .routing import run_routing; run_routing()`

That's it. Help text auto-updates, and the example is runnable.

## Configuration & Secrets

**Environment Variables** (see `.env.example`):
- `AWS_REGION` — Bedrock region (default: ap-southeast-2)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — Required for Bedrock
- `AWS_SESSION_TOKEN` — Optional (for temporary credentials)
- `BEDROCK_MODEL_ID` — Model to use (default: amazon.nova-micro-v1:0)

**Validation**: `src/app/config.py:validate_aws_credentials()` checks for required keys and raises `ValueError` if missing.

## Key Patterns

### LangChain + Bedrock

All examples use `langchain_aws.ChatBedrock` for LLM calls:
```python
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id=BEDROCK_MODEL_ID, region_name=AWS_REGION)
```

### Prompt Chaining Example (`src/examples/prompt_chaining/spec_extractor.py`)

Two-step pipeline:
1. **Extraction**: Raw text → LLM extracts specs (string)
2. **Transformation**: Specs → LLM transforms to JSON

Uses LangChain pipe operator (`|`) to chain runnables. The lambda step (specs string → dict) is crucial—it reformats data for the next prompt.

Demonstrates **best practice for structured output**: validate JSON with `json.loads()` and handle `JSONDecodeError`. Relying on "just ask the model for JSON" fails in production.

### Shared Utilities

`src/examples/cli.py` is the pattern for shared code across examples:
- Centralized example registry
- Reusable CLI helpers
- Easy to extend without duplicating logic

## Common Tasks

**Run an example**: `uv run python -m src.examples prompt_chaining`

**Show all examples**: `uv run python -m src.examples help`

**Add a new example**: Follow "Adding New Examples" above. Most of the boilerplate is the `__init__.py` and CLI registration; the actual logic lives in your module.

**Modify an example**: Examples are in `src/examples/[name]/`. Each is standalone—edit the module and re-run.

**Debug imports**: If you see `ModuleNotFoundError`, ensure:
- Dependencies synced: `uv sync`
- Relative imports use correct depth (e.g., `from ...app.config` from deep in examples)
- Python module structure is correct (`__init__.py` files present)

## Testing Strategy

*Currently no test suite.* If tests are added:
- Place in `tests/` at project root
- Run with `uv run pytest` (after adding pytest to dependencies)
- Examples should be testable as standalone functions (support dependency injection for mocking LLM)

## Dependencies

Core:
- **langchain** — LangChain framework
- **langchain-aws** — AWS Bedrock integration  
- **python-dotenv** — Environment variable loading

See `pyproject.toml` for versions and full list.

## Files You'll Edit

**Frequently:**
- `src/examples/*/` — Adding/modifying examples
- `src/examples/cli.py` — Registering new examples
- `src/examples/__main__.py` — Adding CLI handlers

**Sometimes:**
- `src/app/config.py` — Adding config options
- `.env` / `.env.example` — Environment secrets

**Rarely:**
- `pyproject.toml` — Adding dependencies (use `uv add`, don't edit manually)
- `app.py` — Entry point logic (stable)
