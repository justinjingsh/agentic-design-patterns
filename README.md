# Agentic Design Patterns

A collection of design patterns and examples demonstrating how to build agentic applications using LangChain and AWS Bedrock.

## Features

- **Prompt Chaining**: Extract technical specifications from product text and transform them into structured JSON format
- **Routing**: Route user requests to different tasks based on intent
- **AWS Bedrock Integration**: Examples using Claude models via AWS Bedrock
- **Structured Output**: Demonstrates best practices for validating LLM outputs

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
│       └── routing/                # Routing example
│           ├── __init__.py
│           └── task_cordinator_agent.py  # Task routing and orchestration
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
