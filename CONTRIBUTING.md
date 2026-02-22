# Contributing to Tileon

Thank you for your interest in contributing to Tileon! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub. When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, GPU, etc.)

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Ensure the code passes all tests
5. Submit a pull request

#### Pull Request Guidelines

- Follow the existing code style
- Write clear commit messages
- Include tests for new features or bug fixes
- Keep pull requests focused on a single change

## Development Setup

### Prerequisites

- Python 3.10+
- CMake 3.20+
- Ninja
- A C++ compiler

### Building from Source

```bash
# Clone the repository
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run Python tests
pytest python/test/
```

## Code Style

We use pre-commit hooks to enforce code style. Please install and run pre-commit before submitting changes:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Getting Help

If you have questions or need help, feel free to open an issue or discussion on GitHub.

## License

By contributing to Tileon, you agree that your contributions will be licensed under the MIT License.
