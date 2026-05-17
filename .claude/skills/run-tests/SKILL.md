# Skill: Run Tests

## Overview
This skill handles running tests for the NeMo project, including unit tests, integration tests, and coverage reporting.

## When to Use
- After making code changes to verify nothing is broken
- Before submitting a PR
- When debugging test failures
- To check code coverage for new features

## Steps

### 1. Identify Test Scope
Determine which tests to run based on the changes made:
- **Full suite**: Run all tests when making broad changes
- **Module-specific**: Run tests for the specific module changed
- **Single test**: Run a specific failing test for debugging

### 2. Set Up Environment
```bash
# Ensure dependencies are installed
pip install -e ".[dev]"

# Verify test dependencies
pip install pytest pytest-cov pytest-xdist
```

### 3. Run Tests

#### Run all tests
```bash
pytest tests/ -v
```

#### Run specific module tests
```bash
# Example: run only ASR tests
pytest tests/collections/asr/ -v

# Example: run only NLP tests
pytest tests/collections/nlp/ -v
```

#### Run with coverage
```bash
pytest tests/ --cov=nemo --cov-report=term-missing --cov-report=html
```

#### Run in parallel (faster)
```bash
pytest tests/ -n auto -v
```

#### Run a single test
```bash
pytest tests/path/to/test_file.py::TestClass::test_method -v
```

### 4. Interpret Results

#### Passing output looks like:
```
========================= X passed in Xs =========================
```

#### Failing output shows:
```
FAILED tests/path/to/test.py::test_name - AssertionError: ...
========================= X failed, Y passed in Zs =========================
```

### 5. Debug Failures

#### Get verbose output for failures
```bash
pytest tests/path/to/failing_test.py -v -s --tb=long
```

#### Run with pdb on failure
```bash
pytest tests/path/to/failing_test.py --pdb
```

#### Check for import errors
```bash
pytest tests/path/to/failing_test.py --collect-only
```

### 6. Coverage Analysis

After running with `--cov`, check:
- `htmlcov/index.html` for detailed HTML report
- Terminal output for quick summary
- `.coveragerc` for excluded files/directories

Target coverage thresholds (from `.coveragerc`):
- New code should maintain or improve existing coverage
- Critical paths (training, inference) should have >80% coverage

## Common Issues

### GPU Tests Skipped
Many NeMo tests require GPU. If running on CPU-only machine:
```bash
# Tests requiring GPU will be automatically skipped
# Look for: SKIPPED (requires GPU)
```

### Import Errors
```bash
# Check if NeMo is properly installed
python -c "import nemo; print(nemo.__version__)"

# Reinstall if needed
pip install -e . --no-build-isolation
```

### Memory Issues
```bash
# Run tests sequentially if OOM
pytest tests/ -v --forked

# Or limit parallel workers
pytest tests/ -n 2
```

### Slow Tests
```bash
# Skip slow tests marked with @pytest.mark.slow
pytest tests/ -m "not slow"

# Run only fast unit tests
pytest tests/unit/ -v
```

## CI/CD Integration
Tests are automatically run on:
- Pull request creation/update
- Push to main/development branches
- Nightly scheduled runs for full suite

Check `.github/workflows/` for CI configuration details.
