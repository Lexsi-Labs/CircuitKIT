# Contributing to CircuitKit

Thank you for your interest in contributing to CircuitKit! This document provides guidelines and instructions for contributing.

## Code of Conduct

CircuitKit is committed to providing a welcoming and inspiring community. We respect all contributors and expect all interactions to be respectful and professional.

## How to Contribute

### Reporting Bugs

Found a bug? Please report it by opening a [GitHub Issue](https://github.com/Lexsi-Labs/circuitkit/issues) with:

1. **Clear description** of the problem
2. **Steps to reproduce** the issue
3. **Expected vs. actual behavior**
4. **Environment details**: Python version, OS, CUDA version (if applicable)
5. **Error messages** and stack traces
6. **Code sample** demonstrating the issue (if applicable)

### Suggesting Features

Have an idea for a feature? Please:

1. Check [existing issues](https://github.com/Lexsi-Labs/circuitkit/issues) to avoid duplicates
2. Open a new issue with:
   - Clear description of the feature
   - Use case and motivation
   - Expected behavior
   - Proposed API (if applicable)
   - Any implementation considerations

### Improving Documentation

Documentation improvements are highly valued:

- Fix typos and clarify existing docs
- Add missing documentation sections
- Create examples and tutorials
- Improve API docstrings
- Add code comments for complex logic

To contribute documentation:

1. Fork the repository
2. Create a branch: `git checkout -b docs/improvement`
3. Make your changes
4. Test that documentation builds: `sphinx-build docs/ build/`
5. Submit a pull request

### Code Contributions

#### Setup Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CircuitKit.git
   cd CircuitKit
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e .[dev,corruption,benchmarks]
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or for bug fixes:
   git checkout -b fix/issue-description
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Follow PEP 8 style guide
   - Add type hints (Python 3.10+ compatible)
   - Include comprehensive docstrings
   - Add unit tests for new functionality

3. **Test your changes**:
   ```bash
   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/unit/test_feature.py
   
   # Run with coverage
   pytest --cov=circuitkit tests/
   ```

4. **Format your code**:
   ```bash
   # Format with black
   black src/circuitkit tests/
   
   # Check import sorting
   isort src/circuitkit tests/
   
   # Lint with flake8
   flake8 src/circuitkit tests/
   ```

5. **Type check**:
   ```bash
   mypy src/circuitkit
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Clear, descriptive commit message"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Open a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Explanation of changes
   - Any breaking changes noted

### Code Style Guide

CircuitKit follows these conventions:

#### Style
- **PEP 8**: Follow [PEP 8](https://pep8.org/) style guide
- **Formatter**: Use `black` with default settings
- **Import sorting**: Use `isort` with default settings
- **Line length**: 100 characters (project default)

#### Type Hints
- Add type hints to all function signatures
- Prefer `Optional[T]` for consistency with the existing codebase
- Document complex types in docstrings

Example:
```python
from typing import Optional, List, Dict
from pathlib import Path

def process_circuits(
    model_name: str,
    scores: Dict[str, float],
    output_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Process discovered circuits.
    
    Args:
        model_name: Name of the transformer model
        scores: Node importance scores
        output_path: Optional path to save results
        
    Returns:
        Dictionary with processed results
    """
    pass
```

#### Documentation
- **Docstrings**: Use Google-style docstrings
- **Module docstrings**: Describe module purpose at the top
- **Class docstrings**: Describe class functionality and attributes
- **Function docstrings**: Include Args, Returns, Raises sections
- **Inline comments**: Explain "why", not "what"

Example:
```python
class CircuitScores:
    """Unified artifact for circuit importance scores.
    
    This class provides a standardized schema for node-level importance
    scores across all 13 circuit discovery algorithms (the EAP family,
    ACDC, IBCircuit, and CD-T).
    
    Attributes:
        task: Name of the task (e.g., 'ioi', 'sva')
        model: Model name (e.g., 'gpt2')
        algorithm: Discovery algorithm used
        node_scores: Dict mapping node names to importance scores
    """
    
    def __init__(self, task: str, model: str, algorithm: str, 
                 node_scores: Dict[str, float]) -> None:
        """Initialize CircuitScores.
        
        Args:
            task: Task name
            model: Model identifier
            algorithm: Algorithm name
            node_scores: Importance scores for each node
            
        Raises:
            ValueError: If scores not in [0, 1] range
        """
        pass
```

### Testing Guidelines

#### Test Location
- Unit tests: `tests/unit/test_*.py`
- Integration tests: `tests/integration/test_*.py`

#### Test Structure
```python
import pytest
from unittest.mock import Mock, patch
from circuitkit.module import MyClass

class TestMyClass:
    """Test suite for MyClass."""
    
    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        return MyClass(param1="value1")
    
    def test_initialization(self, setup):
        """Test class initialization."""
        assert setup.param1 == "value1"
    
    def test_method_with_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError):
            MyClass(param1=None)
```

#### Coverage Requirements
- Aim for >80% code coverage
- Test both happy path and error cases
- Include edge case tests
- Mock external dependencies

#### Running Tests
```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_feature.py

# Specific test class
pytest tests/unit/test_feature.py::TestMyClass

# Specific test method
pytest tests/unit/test_feature.py::TestMyClass::test_method

# With coverage report
pytest --cov=circuitkit --cov-report=html tests/

# Verbose output
pytest -v tests/

# Stop on first failure
pytest -x tests/
```

### Git Workflow

#### Commit Messages
- Use imperative mood ("Add feature" not "Added feature")
- First line should be ≤50 characters
- Add detailed explanation in body (if needed)
- Reference issues: "Fixes #123" or "Relates to #456"

Example:
```
Add CircuitScores JSON serialization

Implement to_json() and from_json() methods with versioning.
Enables cross-platform artifact exchange. Fixes #234.
```

#### Pull Requests
- One feature per PR (atomic changes)
- Keep PRs focused and manageable
- Reference related issues
- Include description of changes
- Request review from maintainers

#### Branches
- Feature: `feature/description` (e.g., `feature/soft-healing`)
- Bug fix: `fix/issue-description` (e.g., `fix/memory-leak`)
- Documentation: `docs/topic` (e.g., `docs/installation-guide`)
- Refactoring: `refactor/area` (e.g., `refactor/evaluation-module`)

### Documentation Build

Build documentation locally:

```bash
pip install -e .[docs]
cd docs
sphinx-build -b html . _build/html
open _build/html/index.html  # View in browser
```

## Areas for Contribution

### High-Priority Areas
- New circuit discovery backends
- Additional evaluation metrics
- Performance optimizations
- Documentation improvements
- Bug fixes

### Medium-Priority Areas
- Additional supported tasks
- Enhanced visualization
- More examples
- Tutorial notebooks
- CLI improvements

### Lower-Priority Areas
- Code style improvements
- Dependency updates
- Refactoring
- Test improvements

## Review Process

### What Reviewers Look For

1. **Code Quality**
   - Follows style guide
   - Type hints present
   - Well-documented
   - No unnecessary complexity

2. **Testing**
   - New tests for new code
   - Tests pass locally
   - Good coverage
   - Edge cases covered

3. **Documentation**
   - Clear docstrings
   - Updated README if needed
   - Examples provided
   - API clearly explained

4. **Functionality**
   - Solves the stated problem
   - No regressions
   - Backward compatible (if applicable)
   - Handles edge cases

### Timeline

- **Small changes** (docs, typos): 1-2 days
- **Bug fixes**: 2-3 days
- **Features**: 1-2 weeks
- **Major features**: Case-by-case

## Development Tips

### Useful Commands

```bash
# Watch tests during development
pytest-watch

# Format on save (if configured)
black --watch src/

# Interactive Python with CircuitKit
python -c "from circuitkit import *; import circuitkit; print(dir())"

# Run specific backend tests
pytest tests/unit -k "backend"

# Profile memory usage
python -m memory_profiler script.py

# Check for security issues
bandit -r src/circuitkit
```

### Common Issues

**Issue**: Tests fail after code changes
- Run `pytest` to check all tests
- Check for breaking changes in public APIs
- Ensure backward compatibility

**Issue**: Import errors in development
- Run `pip install -e .` to reinstall editable version
- Check Python path: `python -c "import sys; print(sys.path)"`

**Issue**: Merge conflicts
- Communicate early with other contributors
- Keep PRs focused and small
- Update from main frequently

## Questions?

- **General**: Open a [Discussion](https://github.com/Lexsi-Labs/circuitkit/discussions)
- **Issues**: Check [existing Issues](https://github.com/Lexsi-Labs/circuitkit/issues)

## Recognition

Contributors are recognized in:
- Git commit history
- GitHub contributors page
- Release notes
- CONTRIBUTORS.md file

## License

By contributing to CircuitKit, you agree that your contributions will be licensed under the Lexsi Labs Source Available License (LSAL) v1.1 (see [LICENSE.md](LICENSE.md), Section 6).

---

**Last Updated**: 2026-04-13  
**Contributing Guidelines Version**: 1.0
