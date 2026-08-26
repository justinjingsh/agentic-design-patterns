# Agentic Design Patterns

A collection of design patterns and examples demonstrating how to build agentic applications using LangChain and AWS Bedrock.

## Features

- **Prompt Chaining**: Extract technical specifications from product text and transform them into structured JSON format
- **Routing**: Route user requests to different tasks based on intent
- **Parallelization**: Run independent LLM sub-tasks (summary, sentiment, keywords) concurrently and merge their results
- **AWS Bedrock Integration**: Examples using Claude models via AWS Bedrock
- **Structured Output**: Demonstrates best practices for validating LLM outputs

## Implementation Approach: LCEL Primitives, not `create_agent`

The examples in this repo are built directly from **LangChain Expression Language (LCEL)** primitives — `Runnable`, the `|` compose operator, `RunnableBranch`, `RunnableLambda`, `RunnableParallel` — rather than from LangChain's prebuilt `create_agent` (a higher-level constructor that wraps an LLM-driven tool-calling loop, formerly `create_react_agent`).

This is intentional: `create_agent` is itself built out of these same LCEL/LangGraph primitives, so writing examples at this lower layer makes each pattern's mechanics visible instead of hiding them behind one call.

- **`prompt_chaining`** (`spec_extractor.py`): a fixed, linear sequence of two LLM calls (extract → transform) composed with `|`. The control flow is deterministic — there's no decision-making about what to do next — so a plain LCEL chain is the right level of abstraction.
- **`routing`** (`routing/cordinator_agent.py`): the LLM classifies the request into a category; a `RunnableBranch` (a plain conditional, not the LLM) dispatches to hand-written Python handler functions. This is "LLM as classifier + deterministic dispatch," which differs from an autonomous tool-calling agent.
- **`parallelization`** (`parallelization/text_analysis.py`): three independent LLM sub-tasks (summary, sentiment, keywords) run concurrently against the same input via `RunnableParallel`, and their results are merged into one report. Because the sub-tasks don't depend on each other's output, fanning them out is safe and cuts wall-clock time to roughly the slowest single sub-task instead of the sum of all three.

**When to reach for `create_agent` instead:** when the LLM itself needs to decide *which* tool(s) to call, possibly in a multi-step loop, reasoning over each result before continuing (e.g., "look up flight prices, then check hotel availability, then answer"). `create_agent` absorbs that observe → decide → act → repeat loop so you don't hand-roll it. It's less suited to cases like this repo's routing example, where dispatch is deterministic and fixed by a lookup map rather than left to the model's judgment.

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
│       ├── prompt_chaining/        # Prompt chaining example
│       │   ├── __init__.py
│       │   └── spec_extractor.py   # Spec extraction and JSON transformation
│       ├── routing/                # Routing example
│       │   ├── __init__.py
│       │   └── task_cordinator_agent.py  # Task routing and orchestration
│       └── parallelization/        # Parallelization example
│           ├── __init__.py
│           └── text_analysis.py    # Concurrent summary/sentiment/keyword analysis
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
