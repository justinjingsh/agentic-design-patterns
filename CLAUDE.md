# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

**Dependencies & Setup**
```bash
uv sync                    # Install/sync all dependencies
uv run app                 # Run main application
uv run python -m src.examples prompt_chaining  # Run prompt chaining example
uv run python -m src.examples routing          # Run rounting example
uv run python -m src.examples parallelization  # Run parallelization example
uv run python -m src.examples reflection       # Run reflection example
uv run python -m src.examples help             # List all available examples
```

**Project is managed with `uv`** — a fast Python package manager. All Python commands should use `uv run` prefix.

## Architecture Overview

### Modular Example-Based Structure

This is a **learning/demo project** organized around design patterns, not a monolithic application:

```
src/
├── app/              # Core application (config, validation, entry point)
│   └── cli.py        # Shared CLI utilities (help text, example registry)
├── examples/         # All design pattern examples (loosely coupled)
│   ├── __main__.py   # CLI dispatcher using dictionary dispatch
│   ├── prompt_chaining/  # Prompt chaining example
│   ├── routing/          # Routing example
│   ├── parallelization/  # Parallelization example
│   └── reflection/       # Reflection example
```

### Key Design Decisions

1. **Examples are First-Class**: Each example is a standalone module under `src/examples/`. They don't import from each other; they only share utilities via `src/app/cli.py`.

2. **Shared CLI Registry**: `src/app/cli.py` defines an `EXAMPLES` dictionary that centralizes:
   - Available examples and their descriptions
   - Help text generation (auto-updates as examples are added)
   - This is the single source of truth for CLI documentation

3. **Entry Point Pattern**: 
   - `app.py` (project root) → wrapper to find `src` module
   - `src.app:main()` → validates AWS credentials, shows usage
   - `src.examples:main()` → CLI dispatcher to run individual examples

4. **Lazy Imports**: Example code imports only when the command is invoked (keeps startup fast, avoids loading unused dependencies).

5. **Dictionary Dispatch**: `src/examples/__main__.py` uses a `COMMANDS` dictionary to map command constants to handler functions, making it easy to add new examples without complex if/elif chains.

## Adding New Examples

**Minimal steps:**

1. Create a new module: `src/examples/new_pattern/module.py` with a `run_*()` function
2. Export your function: `src/examples/new_pattern/__init__.py` → `from .module import run_new_pattern; __all__ = ["run_new_pattern"]`
3. Register in `src/app/cli.py` → add `CMD_NEW_PATTERN = "new_pattern"` and `EXAMPLES[CMD_NEW_PATTERN] = "description"`
4. Add handler in `src/examples/__main__.py`:
   - Create helper: `def _run_new_pattern() -> None: ... from .new_pattern import run_new_pattern; run_new_pattern()`
   - Register in `COMMANDS` dict: `CMD_NEW_PATTERN: _run_new_pattern,`

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

### Logging Best Practices

Use lazy % formatting instead of f-strings for efficiency (the logger only formats if the log level is enabled):

```python
# Good: lazy formatting
logger.info("Processing example: %s", example_name)
logger.error("Configuration error: %s", e)

# Avoid: f-string formatting
logger.info(f"Processing example: {example_name}")  # Formats even if INFO is disabled
```

The project has `import-outside-toplevel` disabled globally in `pyproject.toml` since lazy imports are intentional for startup speed.

### Prompt Chaining Example (`src/examples/prompt_chaining/spec_extractor.py`)

Two-step pipeline:
1. **Extraction**: Raw text → LLM extracts specs (string)
2. **Transformation**: Specs → LLM transforms to JSON

Uses LangChain pipe operator (`|`) to chain runnables. The lambda step (specs string → dict) is crucial—it reformats data for the next prompt.

Demonstrates **best practice for structured output**: validate JSON with `json.loads()` and handle `JSONDecodeError`. Relying on "just ask the model for JSON" fails in production.

### Parallelization Example (`src/examples/parallelization/text_analysis.py`)

Three independent sub-tasks (summary, sentiment, keywords) run concurrently against the same input:

1. Each sub-task is its own linear chain: `prompt | llm | StrOutputParser()`.
2. `RunnableParallel(summary=..., sentiment=..., keywords=...)` fans all three out at once (via a thread pool) instead of running them sequentially, and merges their outputs into a single dict keyed by the names passed to it.

Unlike `prompt_chaining`, the sub-tasks here don't depend on each other's output, which is what makes concurrent execution safe. `analyze_text()` times the call to show that wall-clock reflects the slowest branch, not the sum of all three.

### Reflection Example (`src/examples/reflection/reflection.py`)

A generate -> reflect -> refine loop, run per task in `run_reflection_loop()`:

1. **Generate**: `build_generate_chain()` produces a first-draft answer to the task.
2. **Reflect**: `build_reflect_chain()` critiques the draft against the task, and outputs exactly `APPROVAL_TOKEN` ("APPROVED") if there's nothing left to fix.
3. **Refine**: `build_refine_chain()` rewrites the draft using that critique — skipped once the reflector approves.

Each chain is `prompt | llm | StrOutputParser()`, rebuilt fresh per call since chains are stateless; the loop's only state is the evolving `draft` string. The loop runs until the reflector's output exact-matches `APPROVAL_TOKEN` or `MAX_ITERATIONS` (3) is reached, whichever comes first — the exact-match check (not a substring check) avoids false positives from a critique that mentions the word while still listing problems.

### Shared Utilities

`src/app/cli.py` provides shared code across examples:
- Centralized example registry (`EXAMPLES` dict)
- Reusable CLI helpers (`print_help`, command constants)
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

Tests exist in `tests/` at project root. Run with `uv run pytest`.

**Test structure:**
- `test_spec_extractor.py` — Tests prompt chaining example with mocked LLM
- `test_config.py` — Tests AWS credential validation
- `test_cli.py` — Tests CLI dispatch logic

**Guidelines:**
- Examples should be testable as standalone functions (support dependency injection for mocking LLM)
- Never import examples at module scope if they touch `src.app.bedrock` (credentials required at import time)
- Tests must not invoke the CLI's `main()` function directly (it validates credentials upfront)

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
