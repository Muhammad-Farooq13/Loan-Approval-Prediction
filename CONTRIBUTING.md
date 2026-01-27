# Contributing to Loan Prediction Project

Thank you for considering contributing to the Loan Prediction Project! This document provides guidelines for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/loan-prediction.git
   cd loan-prediction
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/original/loan-prediction.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 isort pre-commit
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Describe the bug clearly with steps to reproduce
- Include error messages and environment details
- Add screenshots if applicable

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement
- Explain why it would be useful
- Provide examples if possible

### Code Contributions

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards

3. Write or update tests for your changes

4. Run tests and linting:
   ```bash
   make test
   make lint
   ```

5. Commit your changes (see commit message guidelines)

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guide
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Maximum line length: 100 characters

### Code Formatting

Format your code before committing:
```bash
make format
# or
black src tests
isort src tests
```

### Linting

Run linting checks:
```bash
make lint
# or
flake8 src tests
```

### Documentation

- Add docstrings to all functions, classes, and modules
- Use Google-style docstrings
- Update README.md if adding new features
- Comment complex logic

Example docstring:
```python
def example_function(param1, param2):
    """
    Brief description of the function.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        
    Returns:
        type: Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features
- Use pytest for testing
- Aim for >80% code coverage
- Test edge cases and error conditions

### Test Structure

```python
def test_function_name():
    """Test description"""
    # Arrange
    input_data = ...
    expected_output = ...
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(models): add XGBoost model implementation

Add XGBoost classifier as an alternative model option
with hyperparameter tuning capabilities.

Closes #123
```

```
fix(preprocessing): handle missing values in categorical columns

Fixed issue where categorical columns with missing values
caused preprocessing pipeline to fail.

Fixes #456
```

## Pull Request Process

1. **Before submitting:**
   - Update documentation
   - Add tests for new features
   - Ensure all tests pass
   - Run linting checks
   - Update CHANGELOG.md if applicable

2. **PR Description:**
   - Clearly describe the changes
   - Reference related issues
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process:**
   - Address reviewer comments
   - Keep the PR focused (one feature/fix per PR)
   - Squash commits if requested
   - Rebase on main if needed

4. **After Approval:**
   - PRs will be merged by maintainers
   - Delete your branch after merge

## Additional Guidelines

### File Organization

- Place new modules in appropriate directories
- Follow existing project structure
- Update `__init__.py` files when adding new modules

### Dependencies

- Add new dependencies to `requirements.txt`
- Pin dependency versions
- Document why the dependency is needed

### Performance

- Consider performance implications
- Profile code for bottlenecks
- Use appropriate data structures and algorithms

### Security

- Never commit sensitive information
- Use environment variables for secrets
- Validate all user inputs
- Follow security best practices

## Questions?

If you have questions, please:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with the "question" label

## Recognition

Contributors will be acknowledged in the project README and release notes.

Thank you for contributing! 🎉
