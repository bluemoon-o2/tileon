| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/bluemoon-o2/tileon/actions/workflows/documentation.yml/badge.svg)](https://tensorplay.cn/tileon/) | [![Wheels](https://github.com/bluemoon-o2/tileon/actions/workflows/wheels.yml/badge.svg)](https://github.com/bluemoon-o2/tileon/actions/workflows/wheels.yml) |

# Tileon

Tileon is a language and compiler for parallel programming. Currently under active development with initial interpreter-only support, it focuses on small-scale experiments and education, providing a simple Python-based programming environment for learning, teaching, and prototyping custom parallel compute kernels.

## Quick Installation

You can install Tileon from source:

```bash
git clone https://github.com/bluemoon-o2/tileon.git
cd tileon

pip install -e .
```

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

### Pre-commit Hooks

We use pre-commit hooks to enforce code style:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## License

Tileon is licensed under the MIT License. See [LICENSE](LICENSE.md) for more information.
