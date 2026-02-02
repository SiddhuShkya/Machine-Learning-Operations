# GitHub Action Workflow - Python Project

A simple Python project demonstrating GitHub Actions workflow with automated testing for basic math operations.

## Project Overview

This project implements fundamental mathematical operations (`add` and `sub`) with comprehensive unit tests. It serves as a template for setting up automated testing pipelines using GitHub Actions.

## Features

- ✅ Add operation: Performs addition of two numbers
- ✅ Subtract operation: Performs subtraction of two numbers
- ✅ Automated unit tests with pytest
- ✅ GitHub Actions CI/CD pipeline integration

## Project Structure

```
.
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   └── math_operations.py       # Math functions (add, subtract)
└── tests/
    ├── __init__.py
    └── test_operations.py       # Unit tests for math operations
```

## Requirements

- Python 3.7+
- pandas
- pytest

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd GitHub-Action-Workflow
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Operations

Import and use the math operations in your Python code:

```python
from src.math_operations import add, sub

result1 = add(5, 3)      # Returns 8
result2 = sub(10, 4)     # Returns 6
```

### Running Tests

Execute the test suite:

```bash
pytest tests/
```

Run tests with verbose output:

```bash
pytest tests/ -v
```

## Test Coverage

The test suite includes:
- **test_add()**: Tests addition with positive, negative, and zero values
- **test_sub()**: Tests subtraction with positive, negative, and zero values

## GitHub Actions

This project includes automated testing via GitHub Actions. On every push and pull request, the workflow will:
1. Install dependencies
2. Run the test suite
3. Report test results