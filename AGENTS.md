# AGENTS.md

Project: learning/demo repo of agentic design patterns using LangChain + AWS Bedrock. See `CLAUDE.md` for the deep-dive guide; `README.md` may still reference old module names (see Architecture below).

## Commands

```bash
uv sync                # install deps (use this before anything; never pip install)
uv run pytest          # run all tests (10 tests; configured in pytest.ini, asyncio auto)
uv run pytest tests/test_spec_extractor.py -q   # run one test file
uv run app             # entry point; validates AWS creds then prints example help
uv run python -m src.examples help             # list examples
uv run python -m src.examples prompt_chaining  # run an example (needs live AWS creds)
uv run python -m src.examples routing          # run the routing example
uv run python -m src.examples parallelization  # run the parallelization example
uv run python -m src.examples reflection       # run the reflection example
uv run python -m src.examples tools            # run the tool use example
uv run python -m src.examples planning         # run the planning example
uv run python -m src.examples multiagent       # run the multi-agent example
```

No linter, formatter, or typecheck config exists. Do not invent one; just run `pytest`.

## Architecture (correct the stale CLAUDE.md/README)

- Example registry + CLI helpers live in `src/app/cli.py` (EXAMPLES dict, `print_help`, `CMD_*` constants) — NOT `src/examples/cli.py`.
- `src/examples/__main__.py` is the example CLI dispatcher using dictionary dispatch; its `main()` calls `validate_aws_credentials()` before running **any** example. Add a new helper function and register it in the `COMMANDS` dict.
- `src/app/config.py` calls `load_dotenv()` at import time; env vars are read as module constants. `src/app/bedrock.py` builds the `ChatBedrock` instance at import time and calls `validate_aws_credentials()` at module scope.
- `src/app/examples.py` is dead legacy code (pre-dates the `src/examples/` package); don't extend it.
- Several example modules were renamed (git mv): `routing/cordinator_agent.py` -> `routing/request_router.py`, `reflection/reflection.py` -> `reflection/draft_refiner.py`, `tools/tool_use.py` -> `tools/tool_calling_agent.py`. Use the new names; CLAUDE.md/README may still show the old ones.
- Every example's entry point is `handle_requests()`, exported from its package `__init__.py` and wired into `src/examples/__main__.py` via a `_run_<name>()` helper + `COMMANDS` dict entry, with a `CMD_<NAME>` constant and description in `src/app/cli.py` `EXAMPLES`.
- `src/examples/routing/request_router.py` demonstrates the Routing pattern: an LLM classifies each request into one `HANDLERS` key (`booker`, `info`) or `unclear`, then a `RunnableBranch` dispatches the untouched request to that plain-Python handler. `build_router()` composes `{"category": classifier_chain, "request": passthrough}` into the branch; `HANDLERS` is the single source of truth (branch conditions are generated from it). The model decides once, then is out of the loop.
- `src/examples/parallelization/text_analysis.py` demonstrates the Parallelization pattern: three independent sub-chains (summary, sentiment, keywords) fanned out via `RunnableParallel` and merged into one report — not a sequential `|` chain like `prompt_chaining`.
- `src/examples/reflection/draft_refiner.py` demonstrates the Reflection pattern: a generate -> reflect -> refine loop over three plain LCEL chains (`build_generate_chain`, `build_reflect_chain`, `build_refine_chain`), each built fresh per call. The loop runs in `run_reflection_loop`, capped at `MAX_ITERATIONS` and exiting early on an exact-match `APPROVAL_TOKEN` from the reflector.
- `src/examples/tools/tool_calling_agent.py` demonstrates the Tool Use / function-calling pattern — NOT a fixed LCEL pipeline. Three `@tool` functions (`calculator`, `get_weather`, `word_count`) are bound via `llm.bind_tools(TOOLS)`; `run_agent()` hand-rolls the observe -> decide -> act loop over a growing `list[BaseMessage]`, appending a `ToolMessage` (keyed by `tool_call_id`) per requested call and re-invoking until the model answers without tool calls or `MAX_STEPS` (5) is hit. Registered as `CMD_TOOLS` / `_run_tools`. Tool errors and unknown tool names are returned as strings, never raised.
- `src/examples/planning/task_planner.py` demonstrates the Planning (plan-and-execute) pattern — also not a fixed pipeline. `run_planner()` runs three stages per goal: `build_planner_chain()` (`prompt | llm | StrOutputParser() | _parse_plan`) decomposes the goal into an ordered `list[str]` capped at `MAX_STEPS` (6) — `_parse_plan()` normalises the model's inconsistent list formatting and truncates rather than raising; a deterministic loop calls `build_executor_chain()` once per step (each call sees the goal, full plan text, and `_format_completed()` prior results); `build_synthesis_chain()` merges the step results into the final answer. The whole plan is committed to before any step runs. Registered as `CMD_PLANNING` / `_run_planning`.
- `src/examples/multiagent/research_team.py` demonstrates the Multi-Agent pattern — also not a fixed pipeline. `SPECIALISTS` maps each name (`researcher`, `analyst`, `writer`) to `(one-line role summary, full system prompt)` in one dict so the supervisor's menu and the agents can't drift. `run_team()` loops up to `MAX_TURNS` (6): `build_supervisor_chain()` reads the goal + `_format_transcript()` and returns one specialist name or `DONE_TOKEN` (`"DONE"`); `_choose_next()` reduces the raw reply to its first word and maps it to a specialist, `DONE_TOKEN`, or `None` (unrecognised -> stop + log). The chosen specialist's chain (built once per name, reused) runs and appends its `(name, message)` pair to the shared transcript. `build_editor_chain()` then always runs one deterministic final pass over the whole transcript. Routing between teammates is decided at runtime from the evolving transcript. Registered as `CMD_MULTIAGENT` / `_run_multiagent`.
- Logging uses lazy % formatting (not f-strings) for efficiency; configured globally in `pyproject.toml` to disable `import-outside-toplevel` pylint warnings (lazy imports are intentional for startup speed).

## Gotchas

- Tests already exist in `tests/` (`test_cli.py`, `test_config.py`, `test_spec_extractor.py`) and pass. CLAUDE.md's "Currently no test suite" is outdated.
- Importing any module that touches `src.app.bedrock` (e.g. `spec_extractor.py`) requires AWS credentials at import time. In tests, mock the chains / LLM and pass mocks to `run_and_validate`; never import examples just to inspect data.
- The examples CLI validates credentials upfront, so examples cannot run offline; tests must not invoke the CLI's `main()`.
- Python is pinned to 3.14 (`.python-version`); `pyproject.toml` requires >=3.12.

## Adding an example

1. `src/examples/<name>/<module>.py` with a `run_*()`/entry function
2. `src/examples/<name>/__init__.py` exporting it
3. Register description in `src/app/cli.py` `EXAMPLES` dict and add `CMD_*` constant
4. Add helper function and register in `COMMANDS` dict in `src/examples/__main__.py`
5. Verify: `uv run pytest`
