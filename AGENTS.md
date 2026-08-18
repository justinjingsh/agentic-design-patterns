# AGENTS.md

Project: learning/demo repo of agentic design patterns using LangChain + AWS Bedrock. See `CLAUDE.md` for the existing deep-dive guide (note: it is slightly stale, see Architecture below).

## Commands

```bash
uv sync                # install deps (use this before anything; never pip install)
uv run pytest          # run all tests (10 tests; configured in pytest.ini, asyncio auto)
uv run pytest tests/test_spec_extractor.py -q   # run one test file
uv run app             # entry point; validates AWS creds then prints example help
uv run python -m src.examples help             # list examples
uv run python -m src.examples prompt_chaining  # run an example (needs live AWS creds)
```

No linter, formatter, or typecheck config exists. Do not invent one; just run `pytest`.

## Architecture (correct the stale CLAUDE.md/README)

- Example registry + CLI helpers live in `src/app/cli.py` (EXAMPLES dict, `print_help`, `CMD_*` constants) — NOT `src/examples/cli.py`.
- `src/examples/__main__.py` is the example CLI dispatcher; its `main()` calls `validate_aws_credentials()` before running **any** example. Add a new `elif command == "<name>"` branch there.
- `src/app/config.py` calls `load_dotenv()` at import time; env vars are read as module constants. `src/app/bedrock.py` builds the `ChatBedrock` instance at import time and calls `validate_aws_credentials()` at module scope.
- `src/app/examples.py` is dead legacy code (pre-dates the `src/examples/` package); don't extend it.

## Gotchas

- Tests already exist in `tests/` (`test_cli.py`, `test_config.py`, `test_spec_extractor.py`) and pass. CLAUDE.md's "Currently no test suite" is outdated.
- Importing any module that touches `src.app.bedrock` (e.g. `spec_extractor.py`) requires AWS credentials at import time. In tests, mock the chains / LLM and pass mocks to `run_and_validate`; never import examples just to inspect data.
- The examples CLI validates credentials upfront, so examples cannot run offline; tests must not invoke the CLI's `main()`.
- Python is pinned to 3.14 (`.python-version`); `pyproject.toml` requires >=3.12.

## Adding an example

1. `src/examples/<name>/<module>.py` with a `run_*()`/entry function
2. `src/examples/<name>/__init__.py` exporting it
3. Register description in `src/app/cli.py` `EXAMPLES` dict
4. Dispatch in `src/examples/__main__.py`
5. Verify: `uv run pytest`
