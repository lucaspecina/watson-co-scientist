# Raul Co-Scientist Tests

This directory contains tests for the Raul Co-Scientist system.

## Directory Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests that verify component interactions
- `scripts/`: Standalone scripts for interactive testing and demonstration
- `data/`: Test data directories
  - `small_dataset/`: Small test dataset
  - `full_dataset/`: More comprehensive test dataset
  - `test_fixtures/`: Specific test fixtures

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/unit/test_knowledge_graph.py
```

To run tests with more detailed output:

```bash
pytest -v
```

## Test Data

The test data directories contain structured data that mimics the production data structure but with controlled content for testing purposes. The `small_dataset` contains a minimal set of data for basic tests, while `full_dataset` provides a more comprehensive dataset for integration tests.

## Test Types

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test interactions between components
3. **Script Tests**: Demonstrate system functionality with specific scenarios

## Adding New Tests

When adding new tests:

1. Place unit tests in `unit/`
2. Place integration tests in `integration/`
3. Place demonstration scripts in `scripts/`
4. Use fixtures from `conftest.py` for common testing resources
5. Follow the existing naming convention: `test_*.py`