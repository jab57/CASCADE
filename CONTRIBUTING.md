# Contributing to CASCADE

Thank you for your interest in contributing to CASCADE! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Getting Started

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/jab57/CASCADE.git
   cd CASCADE
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv env

   # Windows
   .\env\Scripts\activate

   # Linux/macOS
   source env/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

   # Install test dependencies
   pip install pytest pytest-cov pytest-asyncio

   # Install PyTorch (CPU for development, GPU for production)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Download data assets:**

   ```bash
   python download_gremln_assets.py
   ```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tools --cov-report=term-missing

# Run a specific test file
pytest tests/test_perturb.py -v
```

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/jab57/CASCADE/issues) to report bugs or suggest features.
- Include steps to reproduce the problem, expected behavior, and actual behavior.
- For bugs, include your Python version, OS, and relevant error messages.

### Submitting Changes

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** Follow the coding standards below.

3. **Add tests** for any new functionality in the `tests/` directory.

4. **Run the test suite** to ensure nothing is broken:

   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes** with a clear, descriptive message:

   ```bash
   git commit -m "Add feature: description of what you did"
   ```

6. **Push and open a Pull Request** against `main`:

   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Guidelines

- Keep PRs focused on a single change or feature.
- Include a description of what the PR does and why.
- Ensure all tests pass before requesting review.
- Add tests for new functionality.
- Update documentation (README, docstrings) if behavior changes.

## Coding Standards

### Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use type hints for function signatures.
- Write docstrings for public functions and classes (Google style).

### Project Structure

- **Tool modules** go in `tools/` — each module should be self-contained with clear inputs/outputs.
- **Tests** go in `tests/` — mirror the module structure (e.g., `tools/perturb.py` → `tests/test_perturb.py`).
- **Workflow logic** lives in `cascade_langgraph_workflow.py`.
- **MCP tool definitions** live in `cascade_langgraph_mcp_server.py`.

### Testing

- Use `pytest` with fixtures defined in `tests/conftest.py`.
- Mock external APIs (Ensembl, STRING, LINCS data files) — tests should not make network calls.
- Mock the GREmLN model checkpoint — tests should run without the 120MB model file.
- Aim for meaningful tests that verify behavior, not just coverage.

## Adding a New Analysis Tool

1. Implement the analysis logic in the appropriate `tools/` module.
2. Add a tool definition in `cascade_langgraph_mcp_server.py` (under `handle_list_tools`).
3. Add a handler case in `handle_call_tool`.
4. Write tests in `tests/`.
5. Update the tool count and table in `README.md`.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code.

## Questions?

Open an issue on GitHub or reach out to the maintainers.
