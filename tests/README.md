# Test Suite

Pipeline-based, organized tests for maintainability and speed.

## Structure

- unit/: fast tests for focused functions/classes (e.g., feature extractors)
- cv/: cross-validation and feature-pipeline tests (subset of real data)
- config/: dotenv-backed configuration system tests
- solid/: SOLID training system tests
- visualization/: plotting/analysis checks (may generate images under logs/)
- smoke/: quick environment/dataset presence checks
- helpers/: shared utilities for tests
- legacy/: archived helper scripts kept for reference (not part of CI)

## How to run (Windows)

- Install pytest (first time only):
```
cmd /c "env\Scripts\activate.bat && pip install pytest"
```

- All tests:
```
cmd /c "env\Scripts\activate.bat && python -m pytest -q"
```

- By suite:
```
cmd /c "env\Scripts\activate.bat && pytest -q tests/unit"
cmd /c "env\Scripts\activate.bat && pytest -q tests/cv"
cmd /c "env\Scripts\activate.bat && pytest -q tests/config"
cmd /c "env\Scripts\activate.bat && pytest -q tests/solid"
cmd /c "env\Scripts\activate.bat && pytest -q tests/visualization"
cmd /c "env\Scripts\activate.bat && pytest -q tests/smoke"
```

## Notes
- Integration-style tests in `tests/cv` use a small subset of `dataset/dataset_periorbital`. If the dataset is missing, tests are skipped.
- Visualization tests may write figures into `logs/`.
- Prefer adding unit tests for new logic and one integration test per pipeline feature.