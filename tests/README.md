# Test Suite

This folder contains a tidy, pipeline-based test organization to keep the project maintainable and easy to execute.

Structure

- unit/: fast tests for small, isolated units (pure functions/classes)
- integration/: tests that exercise multiple modules together using small in-memory or tiny on-disk data
- smoke/: very quick end-to-end sanity checks that run on tiny samples
- test_runner.py: discover and run all tests

How to run

From the project root on Windows (ensuring the venv is active):

```
cmd /c "env\Scripts\activate.bat && python -m pytest -q"
```

Run by group:

```
# unit only
cmd /c "env\Scripts\activate.bat && pytest -q tests/unit"

# integration only
cmd /c "env\Scripts\activate.bat && pytest -q tests/integration"

# smoke only
cmd /c "env\Scripts\activate.bat && pytest -q tests/smoke"
```

Notes

- Tests use a small subset of the dataset at `dataset/dataset_periorbital`. If not present, some integration/smoke tests will skip.
- Configuration is loaded via `.env` through `ml_training.core.config` where relevant.
- Keep unit tests fast (<1s each) and deterministic.