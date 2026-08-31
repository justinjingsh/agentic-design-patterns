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
uv run python -m src.examples tools            # Run tool use example
uv run python -m src.examples planning         # Run planning example
uv run python -m src.examples multiagent       # Run multi-agent collaboration example
uv run python -m src.examples state            # Run state management example
uv run python -m src.examples goal_monitoring  # Run goal setting and monitoring example
uv run python -m src.examples exception_handling  # Run exception handling example
uv run python -m src.examples hitl             # Run human-in-the-loop example
uv run python -m src.examples rag              # Run retrieval-augmented generation example
uv run python -m src.examples a2a              # Run agent-to-agent (A2A) collaboration example
uv run python -m src.examples resource_optimization  # Run resource-aware optimization example
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
│   ├── core_patterns/    # Pattern example modules live here
│   │   ├── prompt_chaining/  # Prompt chaining example
│   │   ├── routing/          # Routing example
│   │   ├── parallelization/  # Parallelization example
│   │   ├── reflection/       # Reflection example
│   │   ├── planning/         # Planning (plan-and-execute) example
│   │   ├── multiagent/       # Multi-agent collaboration example
│   │   └── tools/            # Tool use (function calling) example
│   ├── state_layers/     # State-layer pattern modules
│   │   ├── state/            # State management example
│   │   └── goal_monitoring/  # Goal setting and monitoring example
│   ├── reliability_layers/   # Reliability-layer pattern modules
│   │   ├── exception_handling/  # Exception handling / error recovery example
│   │   ├── hitl/                # Human-in-the-loop approval gate example
│   │   └── rag/                 # Retrieval-augmented generation example
│   └── production_patterns/  # Production-pattern modules
│       ├── a2a/                 # Agent-to-Agent protocol (discovery + task delegation) example
│       └── resource_optimization/  # Resource-aware optimization (tiered cost/quality budget) example
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

1. Create a new module: `src/examples/core_patterns/new_pattern/module.py` with a `run_*()` function
2. Export your function: `src/examples/core_patterns/new_pattern/__init__.py` → `from .module import run_new_pattern; __all__ = ["run_new_pattern"]`
3. Register in `src/app/cli.py` → add `CMD_NEW_PATTERN = "new_pattern"` and `EXAMPLES[CMD_NEW_PATTERN] = "description"`
4. Add handler in `src/examples/__main__.py`:
   - Create helper: `def _run_new_pattern() -> None: ... from .core_patterns.new_pattern import run_new_pattern; run_new_pattern()`
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

### Prompt Chaining Example (`src/examples/core_patterns/prompt_chaining/spec_extractor.py`)

Two-step pipeline:
1. **Extraction**: Raw text → LLM extracts specs (string)
2. **Transformation**: Specs → LLM transforms to JSON

Uses LangChain pipe operator (`|`) to chain runnables. The lambda step (specs string → dict) is crucial—it reformats data for the next prompt.

Demonstrates **best practice for structured output**: validate JSON with `json.loads()` and handle `JSONDecodeError`. Relying on "just ask the model for JSON" fails in production.

### Parallelization Example (`src/examples/core_patterns/parallelization/text_analysis.py`)

Three independent sub-tasks (summary, sentiment, keywords) run concurrently against the same input:

1. Each sub-task is its own linear chain: `prompt | llm | StrOutputParser()`.
2. `RunnableParallel(summary=..., sentiment=..., keywords=...)` fans all three out at once (via a thread pool) instead of running them sequentially, and merges their outputs into a single dict keyed by the names passed to it.

Unlike `prompt_chaining`, the sub-tasks here don't depend on each other's output, which is what makes concurrent execution safe. `analyze_text()` times the call to show that wall-clock reflects the slowest branch, not the sum of all three.

### Routing Example (`src/examples/core_patterns/routing/request_router.py`)

"LLM as classifier + deterministic dispatch." An LLM classifies each request into exactly one of the `HANDLERS` keys (`booker`, `info`) or `unclear`; a `RunnableBranch` then dispatches the *untouched* request text to the plain-Python handler for that category. The model makes one decision and is then out of the loop.

`build_router()` composes a dict-of-runnables `{"category": classifier_chain, "request": passthrough}` (both entries get the same original input) into a `RunnableBranch`. `HANDLERS` is the single source of truth — the branch conditions are generated from it, so adding a category means adding one dict entry. The branch lambdas bind their loop variables via default args (`category=category`, `handler=handler`) to dodge the late-binding closure bug. Handlers take only the raw request string, so routing and handling are fully decoupled.

### Planning Example (`src/examples/core_patterns/planning/task_planner.py`)

The plan-and-execute pattern: split "decide what to do" from "do it." `run_planner()` runs three stages per goal:

1. **Plan**: `build_planner_chain()` (`prompt | llm | StrOutputParser() | _parse_plan`) turns the goal into an ordered `list[str]` of at most `MAX_STEPS` (6) steps. `_parse_plan()` normalises the model's inconsistent list formatting (`1.`, `1)`, `- `, `Step 1:`) and truncates to `MAX_STEPS` rather than raising.
2. **Execute**: a deterministic loop calls `build_executor_chain()` once per step. Every call also sees the goal, the full plan text (formatted once), and `_format_completed()` — the `(step, result)` pairs done so far.
3. **Synthesise**: `build_synthesis_chain()` merges all step results into the final answer.

The entire plan is committed to *before* any step runs, unlike the `tools` example where the model picks the next step after seeing the last result. The loop's only state is `completed`. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_PLANNING` / `_run_planning`.

### Multi-Agent Example (`src/examples/core_patterns/multiagent/research_team.py`)

Supervisor-routed collaboration on a shared transcript. `SPECIALISTS` maps each name to `(one-line role summary, full system prompt)` — one dict so the supervisor's menu and the agents can't drift. `run_team()` loops up to `MAX_TURNS` (6):

1. `build_supervisor_chain()` reads the goal + `_format_transcript()` and replies with one specialist name or `DONE_TOKEN`. `_choose_next()` reduces the raw reply to its first word, lower-cased, and maps it to a specialist, `DONE_TOKEN`, or `None` (unrecognised → stop and log).
2. The chosen specialist's chain (built once per name, reused across turns) runs against the same goal + transcript and appends its `(name, message)` pair.
3. On `DONE_TOKEN`, `None`, or hitting `MAX_TURNS`, the loop ends.

`build_editor_chain()` then always runs one deterministic final pass over the whole transcript, so output shape is predictable even if the team wandered. Routing between teammates is decided at runtime from the evolving transcript — that's what makes it collaboration, not a fixed fan-out like `parallelization`. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_MULTIAGENT` / `_run_multiagent`.

### Reflection Example (`src/examples/core_patterns/reflection/draft_refiner.py`)

A generate -> reflect -> refine loop, run per task in `run_reflection_loop()`:

1. **Generate**: `build_generate_chain()` produces a first-draft answer to the task.
2. **Reflect**: `build_reflect_chain()` critiques the draft against the task, and outputs exactly `APPROVAL_TOKEN` ("APPROVED") if there's nothing left to fix.
3. **Refine**: `build_refine_chain()` rewrites the draft using that critique — skipped once the reflector approves.

Each chain is `prompt | llm | StrOutputParser()`, rebuilt fresh per call since chains are stateless; the loop's only state is the evolving `draft` string. The loop runs until the reflector's output exact-matches `APPROVAL_TOKEN` or `MAX_ITERATIONS` (3) is reached, whichever comes first — the exact-match check (not a substring check) avoids false positives from a critique that mentions the word while still listing problems.

### Tool Use Example (`src/examples/core_patterns/tools/tool_calling_agent.py`)

The Tool Use (function calling) pattern — the first example whose control flow is **not** a fixed LCEL pipeline. The model decides at runtime which tools to call and how many turns to take.

1. **Tools**: three plain functions wrapped with `@tool` (`calculator`, `get_weather`, `word_count`). The decorator turns each function's name, docstring, and type-hinted signature into the JSON schema the model sees — so the docstring is written for the model, not as an internal comment. `calculator` sandboxes `eval` behind a character whitelist plus `{"__builtins__": {}}`; `get_weather` reads a hard-coded offline dict so the example is deterministic. Every tool returns a `str`. `TOOLS` / `TOOLS_BY_NAME` are both derived from one list so they can't drift.
2. **Loop** (`run_agent()`): `llm.bind_tools(TOOLS)` attaches the schemas (it does not execute anything). The conversation is a growing `list[BaseMessage]` — `SystemMessage`, `HumanMessage`, then alternating `AIMessage` (may carry `.tool_calls`) and `ToolMessage` (our results). Each iteration re-sends the whole list. If an `AIMessage` has no `tool_calls`, that's the final answer and the loop returns. Otherwise every requested call is run, and each result is appended as a `ToolMessage` tagged with the matching `tool_call_id` (critical when several tools run in one turn). Unknown tool names and tool errors are returned as error strings rather than raised, so the model can recover.
3. **Bounds**: `MAX_STEPS` (5) caps model invocations; hitting it returns a sentinel string instead of raising, so `handle_requests()` continues with the remaining `SAMPLE_QUERIES`. Each query is an independent run with a fresh message list.

This is the case where LangChain's `create_agent` would normally be used; the loop is hand-rolled here so the observe -> decide -> act -> repeat mechanics are visible.

### State Management Example (`src/examples/state_layers/state/state_manager.py`)

The State Management pattern: keep an explicit, structured state object and thread that *same object* through every turn, instead of re-sending one ever-growing text blob. The step sequence is fixed (unlike `tools`/`planning`); the point of the example is how information is **retained, compressed, and promoted** between three memory tiers.

`ConversationState` (a `@dataclass`) holds:
1. `recent: list[tuple[str, str]]` — short-term memory, the last few turns kept **verbatim**. Capped at `2 * MAX_RECENT_TURNS` (3) entries.
2. `summary: str` — compressed history. A turn evicted from `recent` is folded into this running prose summary, not discarded, so an arbitrarily long conversation stays at roughly fixed token cost.
3. `facts: dict[str, str]` — long-term memory. Durable `key: value` facts promoted out of each exchange; always injected into context regardless of turn age.

`run_conversation()` builds the three chains once (they're stateless — all continuity is in `state`) and runs a four-step cycle per turn:
- **RETRIEVE**: `_format_facts()` / `_format_recent()` flatten `state` into prompt vars (`facts`, `summary`, `recent`, `user`).
- **RESPOND**: `build_respond_chain()` writes the reply from that context; it never sees the raw state.
- **RECORD**: append `("user", msg)` and `("assistant", reply)` to `state.recent`.
- **UPDATE**: `build_fact_chain()` emits `key: value` lines or exactly `NO_FACTS_TOKEN` (`"NONE"`); `_merge_facts()` parses it leniently (skips malformed lines, `partition(":")`, overwrites on repeat so a later correction wins). Then `_compress_recent()` `while`-pops `state.recent` back down to the cap, calling `build_summary_chain()` once per evicted turn.

`SAMPLE_CONVERSATION` is scripted so later turns depend on earlier ones: by the final "quick recap" turn the first messages have slid out of `recent` and survive only in `summary` / `facts`, so a correct answer there proves the retention logic works. Entry point `handle_requests()` (exported from `__init__.py`) also prints the final state. Registered as `CMD_STATE` / `_run_state`.

### Goal Setting and Monitoring Example (`src/examples/state_layers/goal_monitoring/goal_tracker.py`)

The Goal Setting and Monitoring pattern: convert a fuzzy goal into an explicit, *measurable* checklist, then work toward it in a loop that checks progress against that checklist after every attempt and stops on an objective condition.

Two halves:
1. **Goal setting** — `build_criteria_chain()` decomposes the goal into a numbered list of atomic, objectively-verifiable success criteria (derived only from what the goal states; required exact strings/placeholders quoted verbatim). `_parse_criteria()` strips list markers and truncates to `MAX_CRITERIA` (6). These criteria — not the original prose — define "done".
2. **Monitoring** — `build_monitor_chain()` scores the current draft criterion-by-criterion, emitting exactly one `<n>. MET` or `<n>. UNMET - <reason>` line per criterion. `_parse_progress()` maps those lines *positionally* onto the criteria (a missing or unparseable line ⇒ UNMET, the safe default so "couldn't confirm" never ends the loop early), producing `list[tuple[criterion, met, reason]]`.

`run_goal_loop()` loops up to `MAX_ITERATIONS` (4): `build_worker_chain()` re-produces the *whole* work product from `(goal, criteria, previous draft, feedback)`; the monitor scores it; if `met_count == len(criteria)` the loop breaks (success), otherwise `_format_feedback()` turns the still-unmet lines into the next attempt's targeted feedback. The `for/else` handles the give-up path — cap hit with criteria still failing, draft + partial progress returned anyway. Loop state is just the `draft` string plus the latest `progress` list.

Contrast: `planning` commits to a fixed ordered step list and runs it once; here there is no step list, the task is re-attempted whole each round, steered only by which criteria still fail. `reflection` uses a free-form prose critic; here the critic is pinned to a fixed, goal-derived checklist so progress is a countable "N of M met". Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_GOAL_MONITORING` / `_run_goal_monitoring`.

### Exception Handling Example (`src/examples/reliability_layers/exception_handling/resilient_agent.py`)

The Exception Handling (error recovery / resilience) pattern: the same hand-rolled tool-calling loop as `tools`, but every tool execution is wrapped in a recovery ladder so a failing tool degrades the answer instead of aborting the run. This is the first module under `reliability_layers/`.

1. **Failure taxonomy**: tools raise `TransientError` (timeout / rate-limit / flaky I/O — a retry might help) or `PermanentError` (bad input, missing resource, auth — a retry won't). `ToolError` is their shared base. Anything else that escapes a tool is caught by a catch-all in `_attempt_with_retries()`, logged with a stack trace, and re-wrapped as `PermanentError`, so callers only ever handle `ToolError` and the loop can't be killed by a surprise exception.
2. **Recovery ladder** in `call_tool_with_recovery()` (never raises, always returns a string):
   - `_attempt_with_retries()` runs the tool, retrying only `TransientError` up to `MAX_RETRIES` (3) times with exponential backoff (`BACKOFF_BASE_SECONDS` (0.5) `* 2 ** (attempt - 1)`); `PermanentError` skips straight past.
   - On unrecoverable failure, if `FALLBACKS` names an alternative tool (`fetch_stock_price` → `fetch_stock_price_backup`), call that instead — transparently, with its own retry budget. The fallback tool is deliberately *not* in `PRIMARY_TOOLS` (the set bound to the model), only in `TOOLS_BY_NAME`, so the model never reasons about which data source is up.
   - Otherwise (or if the fallback also fails) `_degraded()` returns a `TOOL_UNAVAILABLE: ...` note (`UNAVAILABLE_PREFIX`). The system prompt tells the model to answer with what it has and flag the gap; results tagged `[fallback: ...]` / `(delayed)` are flagged as possibly stale.
3. **Deterministic flakiness**: the offline tools script their failures via a module-level `_price_calls` counter (`fetch_stock_price` "times out" `_TRANSIENT_FAILURES` (2) times per ticker, then succeeds), so every run fails and recovers in the same places. The three `SAMPLE_QUERIES` each drive one branch: retry-then-succeed (ADP), permanent-then-fallback (GLOBEX, absent from the primary feed), and unexpected-error-then-degrade (`get_market_news` raises a bare `RuntimeError`, no fallback). `_price_calls` intentionally persists across queries — once a ticker is "warmed up" the feed keeps succeeding.

`run_agent()` is structurally identical to the `tools` example's — a bounded observe → decide → act loop over a growing message list — except tool calls go through `call_tool_with_recovery()`. Contrast: `reflection` / `goal_monitoring` recover from a *low-quality* result by iterating; here the concern is a step that *fails outright* — no output to critique, just an exception to classify and route. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_EXCEPTION_HANDLING` / `_run_exception_handling`.

### Human-in-the-Loop Example (`src/examples/reliability_layers/hitl/approval_agent.py`)

The Human-in-the-Loop pattern: the same hand-rolled tool-calling loop as `tools`, but a **fixed policy** intercepts every *side-effecting* tool call and pauses for a human decision before it runs. Second module under `reliability_layers/`.

1. **Review gate**: `SENSITIVE_TOOLS` (`book_flight`, `send_email`, `cancel_booking`) — the ones that spend money, send external mail, or are irreversible — are never executed directly. `search_flights` / `get_fare_rules` are read-only and run automatically. Membership is a policy set, not the model's call.
2. **The `Reviewer`**: a one-method protocol (`decide(*, tool_name, args, prompt) -> ReviewDecision`). `ReviewDecision.action` is `APPROVE` (run as proposed), `EDIT` (run with `edited_args` instead), `REJECT` (don't run; `message` is the reason), or `ANSWER` (reply to a `request_human_input` question). Two implementations: `ScriptedReviewer` replays canned decisions keyed by tool name (FIFO) so the CLI runs unattended and deterministically — same idea as `exception_handling`'s scripted flakiness; `ConsoleReviewer` blocks on `input()` for a real person and is *not* wired into `handle_requests()` (swap it in via `run_agent(query, ConsoleReviewer())`). Exhausting a script falls back to the safe default: deny a sensitive call, tell the model to use its judgement on a question.
3. **`request_human_input`** is a real bound tool the model can choose when a request is under-specified; the loop special-cases its "execution" to `reviewer.decide(...)` and feeds the answer back as `HUMAN_RESPONSE: ...`.
4. **`_dispatch_call(call, reviewer)`** (always returns a string — a rejection is a *result*, not an error) routes each call: `ASK_HUMAN_TOOL` → reviewer → `HUMAN_RESPONSE: ...`; non-sensitive → run now; sensitive → reviewer, then `REJECT` → `HUMAN_REJECTED: <reason>` (system prompt tells the model not to retry or route around it), `EDIT` → run with the human's args, prefixed `HUMAN_EDITED: ...`, `APPROVE` → run with original args. The prefixes are conventions the system prompt explains, nothing parses them.

`run_agent(query, reviewer)` is structurally identical to the `tools` loop; only the per-call dispatch changed. `SAMPLE_SESSIONS` pairs each request with the script its reviewer replays, one per branch: approve a booking, edit an email's args before sending, reject a cancellation, and clarify an under-specified request via `request_human_input`. Contrast: `exception_handling` also wraps each tool call, but to recover from one that *fails* — here the call would succeed and the point is that it shouldn't happen without a person; `reflection` / `goal_monitoring` put an automated critic *after* the output, here a human gates *before* the action. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_HITL` / `_run_hitl`.

### Retrieval-Augmented Generation Example (`src/examples/reliability_layers/rag/rag_pipeline.py`)

The RAG (Knowledge Retrieval) pattern: before the model answers, a deterministic retrieval step pulls the most relevant passages out of a knowledge base and injects them into the prompt; the model is instructed to answer *only* from those passages, cite each claim with a `[n]`, and emit exactly `INSUFFICIENT_CONTEXT_TOKEN` ("INSUFFICIENT_CONTEXT") when the sources don't hold the answer. Third module under `reliability_layers/` — grounding is treated as a reliability property (faithfulness, attribution, honest refusal, freshness), not just a convenience.

Fixed three-stage pipeline in `run_rag()` — the model makes no control-flow decisions:

1. **RETRIEVE** — `retrieve()` scores every `Document` in `KNOWLEDGE_BASE` by lexical overlap (`_score()` = count of distinct query terms appearing in the doc, title terms counted twice), keeps the top `MAX_CONTEXT_DOCS` (3) whose score clears `MIN_RETRIEVAL_SCORE` (1), ties broken on `doc_id`. Zero hits short-circuits to a refusal with no LLM call. `_tokenize()` lower-cases, splits on `[a-z0-9]+`, and drops `_STOPWORDS` plus 1-char tokens. This lexical retriever stands in for embeddings + a vector store; the retrieve → augment → generate shape is what the example is about.
2. **AUGMENT** — `_format_sources()` renders the retrieved passages as a numbered `[n] title\ntext` block. Those `[n]` labels are what the model cites and what `_check_citations()` validates against.
3. **GENERATE** — `build_answer_chain()` (`prompt | llm | StrOutputParser()`, stateless, rebuilt per run) answers from that block. An exact-match check on `INSUFFICIENT_CONTEXT_TOKEN` (not substring, same rationale as `reflection`'s `APPROVAL_TOKEN`) routes the refusal path. `_check_citations()` then returns any cited `[n]` outside the retrieved range — a grounding smell that's logged and printed but not hard-failed (lenient parsing, like `planning`'s `_parse_plan`).

`KNOWLEDGE_BASE` is a hard-coded list of short product-doc `Document`s; `SAMPLE_QUESTIONS` drive one outcome each: single-doc answer, multi-doc synthesis (answer should cite two sources), and out-of-scope (retrieval weak/irrelevant → `INSUFFICIENT_CONTEXT`). Contrast: `tools` lets the *model* decide to fetch data and with what args; here retrieval is a fixed pre-step. `exception_handling` / `hitl` wrap a tool loop to recover from failure / gate a side effect; RAG has no tool loop — it grounds a single-shot answer and declines when grounding is missing. `reflection` / `goal_monitoring` iterate on a weak answer; RAG changes what the model is *given* before it writes. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_RAG` / `_run_rag`.

### Agent-to-Agent Example (`src/examples/production_patterns/a2a/a2a_orchestrator.py`)

The Agent-to-Agent (A2A) pattern: capabilities live in independent *services*, each publishing a machine-readable **Agent Card** (`name`, `description`, `skills`). A client **orchestrator** discovers those cards at runtime, dispatches to the one whose skills fit the request, and delegates by sending a `Message` over a transport; the remote agent runs the job as a `Task` with an explicit lifecycle and returns structured `history` + `artifacts`. First module under `production_patterns/`.

- **Protocol vocabulary**: `Task` states are plain string constants — `SUBMITTED` → `WORKING` → `COMPLETED` | `INPUT_REQUIRED` | `FAILED`. `AgentCard`, `Message` (`role` = `"user"`/`"agent"`), `Artifact` (named string), and `HandlerResult` (exactly one of `answer` → COMPLETED or `question` → INPUT_REQUIRED; a raise → FAILED) are all `@dataclass`es.
- **Remote-agent side**: `RemoteAgentService` wraps a card + a `Handler` (`str -> HandlerResult`) and drives the lifecycle in `execute()` — it prints every state transition and *never raises* (a failed task is a return value). Two demo agents, each grounding a `_phrase_chain()` LLM call in a hard-coded table: `_fx_handler` (`_RATES_TO_USD`, `_find_amount`/`_find_currencies` lenient parsing; computes the number in Python, LLM only phrases it; asks `INPUT_REQUIRED` when amount or a second currency is missing) and `_weather_handler` (`_FORECASTS` by city; `INPUT_REQUIRED` when no known city is named).
- **Registry + transport**: `AgentRegistry` (`register` / `discover` → cards only / `connect` → `RemoteConnection`) stands in for a discovery service; `RemoteConnection.send(message, task=None)` is the wire — a direct call here, an HTTP POST in production. The orchestrator only ever touches cards and the returned `Task`.
- **Orchestrator side** (`run_orchestrator`): `build_dispatch_chain()` picks one card name or exactly `NO_AGENT_TOKEN` (`"NONE"`); `_first_token()` normalises the reply and an unknown name routes to `_decline()`. `build_request_chain()` crafts the `Message` to send. On `INPUT_REQUIRED`, `build_followup_chain()` answers the agent's question (picking a stated default when the user never supplied the detail) and re-sends against the **same** `task_id`, up to `MAX_ROUNDS` (3). On `COMPLETED`, `build_synthesis_chain()` relays the result. Loop state is just the current `Task` handle and the last `Message`.
- **`SAMPLE_REQUESTS`** drive one path each: clean delegation (fx) → completed; a different agent (weather) → completed; under-specified ("Convert 300 into Japanese yen") → `INPUT_REQUIRED` → orchestrator supplies `USD` default → completed; and a request no card covers → orchestrator declines.

Contrast: `multiagent` routes between personas sharing one in-process transcript; A2A puts each agent behind a card + transport and delegates a task envelope that can pause for input. `routing` classifies once then dispatches to a *local* handler; A2A's target is a remote service with its own lifecycle. `tools` calls in-process functions whose schemas the model knows up front; A2A *discovers* capabilities from cards at runtime. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_A2A` / `_run_a2a`.

### Resource-Aware Optimization Example (`src/examples/production_patterns/resource_optimization/resource_optimizer.py`)

The Resource-Aware Optimization pattern: the agent is given an explicit, finite **budget** and must decide, per unit of work, not just *whether* to do it but *how expensively* — walking a fixed cost/quality ladder down as the remaining budget shrinks, and funding required work before optional work. Second module under `production_patterns/`.

- **Tier ladder** (plain string constants, tried richest-first): `PREMIUM` (full LLM call, thorough multi-sentence prompt) → `STANDARD` (LLM call, one-sentence prompt — the cost lever is prompt verbosity, not a different model, since the repo has one shared `llm`) → `ECONOMY` (no LLM call — a cached fact from `_ECONOMY_CACHE`) → `PLACEHOLDER` (free static note; **required** sections only, so one is never silently dropped) → `SKIP` (section omitted; **optional** sections only).
- **`select_tier(costs, remaining, required)`** returns the first paid tier (`PREMIUM`/`STANDARD`/`ECONOMY`) whose cost fits `remaining`, else `PLACEHOLDER` for a required section or `SKIP` for an optional one.
- **`Budget`** (`@dataclass`: `total`, `spent`, `remaining` property, `spend()`) is the only state threaded through `run_report()`. `SUBTASKS` (a `list[Subtask]`) is listed required-first, and `run_report()` re-sorts on `not s.required` so required sections are always priced and funded before optional ones regardless of list order.
- **`run_report(company, budget_total)`**: for each section, pick a tier, run it (`run_subtask()`), deduct its cost, and print the decision. After all sections, if `budget.remaining >= SYNTHESIS_COST` an LLM pass (`build_synthesis_chain()`) merges the sections into one brief; otherwise synthesis itself degrades to a plain concatenation (`_format_sections()`), no further LLM spend.
- **`SAMPLE_RUNS`** hold the company and task fixed and vary only the budget, so one report demonstrates the whole ladder as spending pressure changes: ample (70 units) → every section at `PREMIUM` plus a real synthesis pass; constrained (10 units) → required sections drop to `STANDARD`/`ECONOMY`, one optional section is skipped, synthesis degrades; minimal (3 units) → required sections hit `ECONOMY` or `PLACEHOLDER`, both optional sections are skipped, synthesis degrades.

Contrast: `exception_handling` also walks a per-step ladder, but it's triggered by a *failure* (retry → fallback → degrade); here every step succeeds and the ladder is chosen *before* the step runs, purely from remaining budget. `hitl` gates side-effecting calls for a *human* to approve; here the gate is automatic and picks a cost tier, not approve/reject. `goal_monitoring` re-attempts the *whole* work product against a checklist until it passes; here the budget is spent once, in one pass, across independent sections — there is no retry, only tier selection. Entry point `handle_requests()` (exported from `__init__.py`), registered as `CMD_RESOURCE_OPTIMIZATION` / `_run_resource_optimization`.

### Shared Utilities

`src/app/cli.py` provides shared code across examples:
- Centralized example registry (`EXAMPLES` dict)
- Reusable CLI helpers (`print_help`, command constants)
- Easy to extend without duplicating logic

## Common Tasks

**Run an example**: `uv run python -m src.examples prompt_chaining`

**Show all examples**: `uv run python -m src.examples help`

**Add a new example**: Follow "Adding New Examples" above. Most of the boilerplate is the `__init__.py` and CLI registration; the actual logic lives in your module.

**Modify an example**: Examples are in `src/examples/core_patterns/[name]/`. Each is standalone—edit the module and re-run.

**Debug imports**: If you see `ModuleNotFoundError`, ensure:
- Dependencies synced: `uv sync`
- Relative imports use correct depth (e.g., `from ....app.config` from a pattern module under `src/examples/core_patterns/<name>/`)
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
- `src/examples/core_patterns/*/` — Adding/modifying examples
- `src/app/cli.py` — Registering new examples (EXAMPLES dict, CMD_* constants)
- `src/examples/__main__.py` — Adding CLI handlers

**Sometimes:**
- `src/app/config.py` — Adding config options
- `.env` / `.env.example` — Environment secrets

**Rarely:**
- `pyproject.toml` — Adding dependencies (use `uv add`, don't edit manually)
- `app.py` — Entry point logic (stable)
