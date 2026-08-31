# Agentic Design Patterns

A collection of design patterns and examples demonstrating how to build agentic applications using LangChain and AWS Bedrock.

## Features

- **Prompt Chaining**: Extract technical specifications from product text and transform them into structured JSON format
- **Routing**: Route user requests to different tasks based on intent
- **Parallelization**: Run independent LLM sub-tasks (summary, sentiment, keywords) concurrently and merge their results
- **Reflection**: Generate a draft, critique it, and refine it in a loop so the model catches its own mistakes
- **Tool Use**: Bind tools (calculator, weather lookup, word count) to the model and run an observe -> decide -> act loop where the model chooses which tools to call
- **Planning**: Have the model write an explicit ordered plan up front, then execute the steps one at a time and synthesise the results
- **Multi-Agent**: A supervisor agent routes a shared transcript between single-purpose teammates (researcher, analyst, writer) until an editor compiles the deliverable
- **State Management**: Thread one structured `ConversationState` through every turn, keeping short-term (verbatim), compressed (rolling summary), and long-term (key/value facts) memory each with its own retention policy
- **Goal Setting and Monitoring**: Turn a fuzzy goal into a checklist of measurable success criteria, then loop work → monitor-against-criteria → targeted retry until every criterion is met or an iteration cap is hit
- **Exception Handling**: Harden the tool-calling loop against failing tools — classify each failure as transient or permanent, retry transient ones with exponential backoff, fall back to an alternative source, and degrade gracefully so one broken tool never aborts the run
- **Human-in-the-Loop**: Gate side-effecting tool calls (book, email, cancel) behind a human who can approve, edit the arguments, or reject — and let the agent ask a clarifying question instead of guessing
- **Retrieval-Augmented Generation**: Ground answers in a retrieved document corpus, cite sources by `[n]`, and decline with a fixed token when the corpus doesn't support an answer
- **Agent-to-Agent (A2A)**: Discover remote agents by a published capability card, delegate a task over a transport with an explicit lifecycle (submitted → working → completed / input-required / failed), and follow up when an agent asks for more information
- **Resource-Aware Optimization**: Spend a finite budget across a task's sub-steps by a tiered cost/quality ladder, funding required work first and degrading gracefully — cheaper prompts, cached facts, or omission — as the budget runs low
- **AWS Bedrock Integration**: Examples using Claude models via AWS Bedrock
- **Structured Output**: Demonstrates best practices for validating LLM outputs

## Implementation Approach: LCEL Primitives, not `create_agent`

The examples in this repo are built directly from **LangChain Expression Language (LCEL)** primitives — `Runnable`, the `|` compose operator, `RunnableBranch`, `RunnableLambda`, `RunnableParallel` — rather than from LangChain's prebuilt `create_agent` (a higher-level constructor that wraps an LLM-driven tool-calling loop, formerly `create_react_agent`).

This is intentional: `create_agent` is itself built out of these same LCEL/LangGraph primitives, so writing examples at this lower layer makes each pattern's mechanics visible instead of hiding them behind one call.

- **`prompt_chaining`** (`spec_extractor.py`): a fixed, linear sequence of two LLM calls (extract → transform) composed with `|`. The control flow is deterministic — there's no decision-making about what to do next — so a plain LCEL chain is the right level of abstraction.
- **`routing`** (`routing/request_router.py`): the LLM classifies the request into a category; a `RunnableBranch` (a plain conditional, not the LLM) dispatches to hand-written Python handler functions. This is "LLM as classifier + deterministic dispatch," which differs from an autonomous tool-calling agent.
- **`parallelization`** (`parallelization/text_analysis.py`): three independent LLM sub-tasks (summary, sentiment, keywords) run concurrently against the same input via `RunnableParallel`, and their results are merged into one report. Because the sub-tasks don't depend on each other's output, fanning them out is safe and cuts wall-clock time to roughly the slowest single sub-task instead of the sum of all three.
- **`reflection`** (`reflection/draft_refiner.py`): a generate -> reflect -> refine loop over three plain LCEL chains. A generator produces a first draft, a reflector critiques it against the task (approving it outright once it's satisfied), and a refiner rewrites the draft using that critique — repeating until approval or a fixed `MAX_ITERATIONS`. The loop itself is hand-written Python, not an agentic tool loop, since the sequence of steps is fixed; only whether to keep iterating is dynamic.
- **`tools`** (`tools/tool_calling_agent.py`): the Tool Use (function calling) pattern. Three `@tool`-decorated functions (`calculator`, `get_weather`, `word_count`) are bound to the model with `llm.bind_tools(...)`. `run_agent()` then drives a hand-written observe -> decide -> act loop: invoke the model over the running message list, execute whatever `tool_calls` it emits, append each result as a `ToolMessage` keyed by `tool_call_id`, and re-invoke — stopping when the model returns an answer with no tool calls or after `MAX_STEPS` (5). Unlike the patterns above, both *which* step runs and *how many* steps run are decided by the model at runtime. The loop is spelled out here (rather than delegated to `create_agent`) so the mechanics are visible.
- **`planning`** (`planning/task_planner.py`): the plan-and-execute pattern. `build_planner_chain()` turns a goal into an explicit ordered list of at most `MAX_STEPS` (6) sub-steps; `run_planner()` then loops deterministically over that plan, calling `build_executor_chain()` once per step (each call also sees the goal, the full plan, and every earlier step's result), and finally `build_synthesis_chain()` merges the per-step results into the answer. The plan is committed to *before* any step runs — "decide what to do" is split from "do it" — which is the opposite of the `tools` loop, where the model picks the next step only after seeing the last result.
- **`multiagent`** (`multiagent/research_team.py`): the Multi-Agent collaboration pattern. Three single-purpose agents (`researcher`, `analyst`, `writer`), each with its own persona, share one transcript. `build_supervisor_chain()` is itself an agent: after every turn it reads the goal and the transcript and names the next speaker, or emits `DONE_TOKEN`. `run_team()` loops that up to `MAX_TURNS` (6), then a deterministic `build_editor_chain()` pass compiles the whole transcript into the deliverable. The routing between teammates is decided at runtime from the evolving shared state, which is what makes this collaboration rather than a fixed fan-out like `parallelization`.
- **`state`** (`state_layers/state/state_manager.py`): the State Management pattern. A single `ConversationState` dataclass (`facts` dict, `summary` string, `recent` list) is threaded through a scripted multi-turn conversation. Each turn runs a fixed four-step cycle — RETRIEVE (flatten state into prompt context), RESPOND (`build_respond_chain()`), RECORD (append the exchange to `recent`), UPDATE (`build_fact_chain()` promotes durable facts into `facts`; `_compress_recent()` evicts turns past `MAX_RECENT_TURNS` and folds each into `summary` via `build_summary_chain()`). The step sequence never varies; what the example shows is how information is *retained, compressed, and promoted* between the three memory tiers so a late turn can still answer from a fact first mentioned before the verbatim window slid past it.
- **`goal_monitoring`** (`state_layers/goal_monitoring/goal_tracker.py`): the Goal Setting and Monitoring pattern. `build_criteria_chain()` turns a goal into a numbered checklist of atomic, objectively-verifiable success criteria (the *goal setting* half). `run_goal_loop()` then loops up to `MAX_ITERATIONS` (4): `build_worker_chain()` (re)produces the whole work product, then `build_monitor_chain()` scores it criterion-by-criterion as `MET` / `UNMET` + reason (the *monitoring* half), `_parse_progress()` turns that into structured verdicts, and the still-unmet lines become the next attempt's feedback. The loop exits the moment every criterion is `MET`, or gives up at the cap and reports which criteria still fail. Unlike `planning` (fixed step list, executed once) the task is re-attempted whole each round; unlike `reflection` (free-form critique) progress is a countable "N of M criteria met" measured against a fixed, goal-derived checklist.
- **`exception_handling`** (`reliability_layers/exception_handling/resilient_agent.py`): the Exception Handling / error-recovery pattern. `run_agent()` is the same bounded observe → decide → act tool-calling loop as `tools`, but each tool call goes through `call_tool_with_recovery()`, which never raises. Tools raise `TransientError` or `PermanentError` (shared base `ToolError`); anything else is caught and re-wrapped as permanent. The ladder: `_attempt_with_retries()` retries only transient failures up to `MAX_RETRIES` (3) with exponential backoff; on an unrecoverable failure the driver consults `FALLBACKS` and transparently calls an alternative tool (the backup is absent from the model-visible `PRIMARY_TOOLS`); if that too fails, `_degraded()` returns a `TOOL_UNAVAILABLE: ...` note the model is told to work around. Failures are scripted (a call counter), so every run recovers in the same places. Where `reflection` / `goal_monitoring` iterate on a *weak* result, this pattern handles a step that *fails outright*.
- **`hitl`** (`reliability_layers/hitl/approval_agent.py`): the Human-in-the-Loop pattern. Same bounded tool-calling loop as `tools`, but each call goes through `_dispatch_call()`, which pauses side-effecting tools (`SENSITIVE_TOOLS` = `book_flight` / `send_email` / `cancel_booking`) for a human before they run; read-only tools run automatically. The human returns a `ReviewDecision` — `APPROVE` (run as proposed), `EDIT` (run with the human's arguments, tagged `HUMAN_EDITED:`), or `REJECT` (never runs, tagged `HUMAN_REJECTED:` — the model is told not to retry or route around it). The agent can also call `request_human_input` to ask a clarifying question instead of guessing. The `Reviewer` is a one-method protocol: `ScriptedReviewer` replays canned decisions so the demo runs unattended and deterministically; `ConsoleReviewer` blocks on `input()` for a real person. Where `exception_handling` wraps a call to recover from *failure*, this gate stops a call that would *succeed* from happening without sign-off.
- **`rag`** (`reliability_layers/rag/rag_pipeline.py`): the Retrieval-Augmented Generation (Knowledge Retrieval) pattern. `run_rag()` is a fixed three-stage pipeline with no model-driven control flow: RETRIEVE (`retrieve()` scores every `Document` in `KNOWLEDGE_BASE` by lexical term overlap — a deterministic stand-in for embeddings + a vector store — and keeps the top `MAX_CONTEXT_DOCS` (3) above `MIN_RETRIEVAL_SCORE` (1); zero hits short-circuits to a refusal with no LLM call), AUGMENT (`_format_sources()` renders them as a numbered `[n]` SOURCES block), GENERATE (`build_answer_chain()` answers *only* from that block, cites each claim `[n]`, or emits exactly `INSUFFICIENT_CONTEXT` when the sources don't hold the answer). A deterministic `_check_citations()` post-check flags any `[n]` outside the retrieved set. The grounding is the reliability property — faithfulness, attribution, and an honest refusal when nothing relevant is retrieved. Unlike `tools`, where the model decides to fetch data and with what arguments, retrieval here is a fixed pre-step; unlike `exception_handling` / `hitl` there is no tool loop at all.
- **`a2a`** (`production_patterns/a2a/a2a_orchestrator.py`): the Agent-to-Agent pattern. Capabilities live behind independent services, each publishing a machine-readable `AgentCard` (name, description, skills). `run_orchestrator()` discovers cards via `AgentRegistry.discover()`, has `build_dispatch_chain()` pick the one card whose skills fit the request (or decline), crafts a `Message` with `build_request_chain()`, and sends it over `RemoteConnection` — a stand-in for an HTTP call. The remote `RemoteAgentService.execute()` drives an explicit `Task` lifecycle (`submitted → working → completed / input-required / failed`) and never raises. On `input-required`, `build_followup_chain()` answers the agent's question and re-sends against the same `task_id`, up to `MAX_ROUNDS` (3); on `completed`, `build_synthesis_chain()` relays the result. Unlike `multiagent`, where personas share one in-process transcript, agents here are discovered by card and invoked across a protocol boundary with their own lifecycle.
- **`resource_optimization`** (`production_patterns/resource_optimization/resource_optimizer.py`): the Resource-Aware Optimization pattern. `run_report()` spends a finite `Budget` across a fixed list of report sections (`SUBTASKS`), required sections funded before optional ones. `select_tier()` picks the richest tier whose cost fits what's left — `PREMIUM` (thorough LLM call) → `STANDARD` (terse LLM call) → `ECONOMY` (a cached fact, no LLM call) — falling back to a free `PLACEHOLDER` note for a required section that can't be afforded at all, or `SKIP` for an optional one. Once every section is filled, an LLM synthesis pass merges them if the budget still allows it, otherwise the sections are concatenated as-is. Unlike `exception_handling`'s ladder, which triggers on a *failure*, this ladder is chosen up front from *remaining budget* — every step succeeds, only its cost and quality vary.

**When to reach for `create_agent` instead:** when the LLM itself needs to decide *which* tool(s) to call, possibly in a multi-step loop, reasoning over each result before continuing (e.g., "look up flight prices, then check hotel availability, then answer"). `create_agent` absorbs that observe → decide → act → repeat loop so you don't hand-roll it — the `tools` example above hand-rolls exactly that loop on purpose, to show what `create_agent` would otherwise hide. It's less suited to cases like this repo's routing example, where dispatch is deterministic and fixed by a lookup map rather than left to the model's judgment.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) - A fast Python package manager
- AWS credentials configured with access to Bedrock models

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd agentic-design-patterns
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**
   
   Create a `.env` file in the project root with your AWS configuration:
   ```env
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   BEDROCK_MODEL_ID=amazon.nova-micro-v1:0
   ```
   
   Or set them as environment variables in your shell.

## Running the Code

### Main Application

```bash
# Run the main app (validates AWS credentials and shows usage)
uv run app
```

### Examples

The project includes organized examples accessible via CLI:

```bash
# Run the prompt chaining example (spec extraction)
uv run python -m src.examples prompt_chaining

# Run the routing example
uv run python -m src.examples routing

# Run the parallelization example
uv run python -m src.examples parallelization

# Run the reflection example
uv run python -m src.examples reflection

# Run the tool use example
uv run python -m src.examples tools

# Run the planning example
uv run python -m src.examples planning

# Run the multi-agent collaboration example
uv run python -m src.examples multiagent

# Run the state management example
uv run python -m src.examples state

# Run the goal setting and monitoring example
uv run python -m src.examples goal_monitoring

# Run the exception handling example
uv run python -m src.examples exception_handling

# Run the human-in-the-loop example
uv run python -m src.examples hitl

# Run the retrieval-augmented generation example
uv run python -m src.examples rag

# Run the agent-to-agent (A2A) collaboration example
uv run python -m src.examples a2a

# Run the resource-aware optimization example
uv run python -m src.examples resource_optimization

# Show available examples and usage
uv run python -m src.examples help
```

### Example Output

When you run the prompt chaining example, it processes product descriptions and extracts specifications:

```
Raw text: Brand new laptop equipped with 5th Gen Intel Core i9 processor (3.1GHz, 8-core), 32GB DDR5 memory, 1TB NVMe SSD storage.

Chain final output (structured JSON, passed validity check):
{
  "cpu": "5th Gen Intel Core i9",
  "memory": "32GB DDR5",
  "storage": "1TB NVMe SSD"
}
```

## Project Structure

```
agentic-design-patterns/
├── src/
│   ├── app/                        # Main application
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── config.py               # AWS configuration and validation
│   │   ├── bedrock.py
│   │   └── examples.py
│   │
│   └── examples/                   # All pattern examples
│       ├── __init__.py
│       ├── __main__.py             # CLI handler for examples
│       └── core_patterns/          # Pattern example modules
│           ├── __init__.py
│           ├── prompt_chaining/    # Prompt chaining example
│           │   ├── __init__.py
│           │   └── spec_extractor.py   # Spec extraction and JSON transformation
│           ├── routing/            # Routing example
│           │   ├── __init__.py
│           │   └── request_router.py   # Classify a request and dispatch to a handler
│           ├── parallelization/    # Parallelization example
│           │   ├── __init__.py
│           │   └── text_analysis.py    # Concurrent summary/sentiment/keyword analysis
│           ├── reflection/         # Reflection example
│           │   ├── __init__.py
│           │   └── draft_refiner.py    # Generate -> reflect -> refine loop
│           ├── planning/           # Planning example
│           │   ├── __init__.py
│           │   └── task_planner.py     # Decompose a goal into steps, then execute them
│           ├── multiagent/         # Multi-agent collaboration example
│           │   ├── __init__.py
│           │   └── research_team.py    # Supervisor-routed researcher/analyst/writer team
│           └── tools/              # Tool use (function calling) example
│               ├── __init__.py
│               └── tool_calling_agent.py  # bind_tools + observe/decide/act loop
│       ├── state_layers/          # State-layer pattern modules
│       │   ├── __init__.py
│       │   ├── state/             # State management example
│       │   │   ├── __init__.py
│       │   │   └── state_manager.py    # One ConversationState threaded through every turn
│       │   └── goal_monitoring/   # Goal setting and monitoring example
│       │       ├── __init__.py
│       │       └── goal_tracker.py     # Goal -> criteria checklist -> work/monitor loop
│       ├── reliability_layers/    # Reliability-layer pattern modules
│       │   ├── __init__.py
│       │   ├── exception_handling/  # Exception handling / error recovery example
│       │   │   ├── __init__.py
│       │   │   └── resilient_agent.py  # retry -> fallback -> graceful degradation ladder
│       │   ├── hitl/                # Human-in-the-loop approval gate example
│       │   │   ├── __init__.py
│       │   │   └── approval_agent.py   # approve / edit / reject gate on side-effecting tools
│       │   └── rag/                 # Retrieval-augmented generation example
│       │       ├── __init__.py
│       │       └── rag_pipeline.py     # retrieve -> augment -> generate, grounded + cited answers
│       └── production_patterns/   # Production-pattern modules
│           ├── __init__.py
│           ├── a2a/                 # Agent-to-Agent protocol example
│           │   ├── __init__.py
│           │   └── a2a_orchestrator.py  # discover cards -> delegate a task -> follow up -> synthesise
│           └── resource_optimization/  # Resource-aware optimization example
│               ├── __init__.py
│               └── resource_optimizer.py  # tiered cost/quality ladder spent against a fixed budget
│
├── app.py                          # Entry point wrapper
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                         # Dependency lock file
└── README.md                       # This file
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `ap-southeast-2` |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_SESSION_TOKEN` | AWS session token (optional) | - |
| `BEDROCK_MODEL_ID` | Model ID to use | `amazon.nova-micro-v1:0` |

### Supported Models

- `anthropic.claude-3-5-sonnet-20241022-v2:0` - Claude 3.5 Sonnet
- `amazon.nova-micro-v1:0` - Amazon Nova Micro
- Other models available via AWS Bedrock

## Dependencies

- **langchain** (>=0.3.0) - LangChain framework
- **langchain-aws** (>=0.2.0) - AWS Bedrock integration
- **python-dotenv** (>=1.2.3) - Environment variable management
- **requests** (>=2.34.2) - HTTP library

See `pyproject.toml` for complete dependency list.

## Adding New Examples

1. Create a new directory under `src/examples/`:
   ```bash
   mkdir src/examples/new_example
   ```

2. Create your example module:
   ```python
   # src/examples/new_example/module.py
   def run_new_example():
       """Your example implementation."""
       pass
   ```

3. Create an `__init__.py` file to export your function:
   ```python
   # src/examples/new_example/__init__.py
   from .module import run_new_example
   __all__ = ["run_new_example"]
   ```

4. Update `src/examples/__main__.py` to add the new command:
   ```python
   # Add helper function
   def _run_new_example() -> None:
       """Run the new example."""
       logger.info("Running new_example")
       from .new_example import run_new_example
       run_new_example()

   # Add to COMMANDS dictionary
   COMMANDS = {
       CMD_PROMPT_CHAINING: _run_prompt_chaining,
       CMD_ROUTING: _run_routing,
       CMD_NEW_EXAMPLE: _run_new_example,
       CMD_HELP: print_help,
   }
   ```

5. Register the command in `src/app/cli.py` and run your example:
   ```bash
   uv run python -m src.examples new_example
   ```

## Troubleshooting

### ModuleNotFoundError

If you get `ModuleNotFoundError`, ensure dependencies are installed:
```bash
uv sync
```

### AWS Credentials Error

Verify your AWS credentials are set correctly:
```bash
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY
```

### Invalid JSON Output

The spec extraction example includes JSON validation. If the LLM output is not valid JSON, you'll see a warning. This demonstrates the importance of validation in production systems.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
