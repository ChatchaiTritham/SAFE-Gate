# Contributing to SAFE-Gate

Thank you for considering a contribution. The following guidelines help
maintain code quality and patient-safety guarantees.

## Getting Started

1. Fork the repository.
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

- Follow PEP 8 conventions.
- Use type hints for function signatures.
- Write docstrings for all public methods (Google style).
- Keep lines under 100 characters where practical.

### Gate Implementation Contract

Every gate class must implement:

| Method              | Returns                          |
|---------------------|----------------------------------|
| `evaluate(data)`    | `(RiskTier, float, dict)`        |
| `get_name()`        | Short identifier (e.g. `"G1"`)   |
| `get_description()` | Human-readable description       |

### Safety Invariants

The following properties **must not** be violated by any code change:

- **Conservative Preservation (CP):** `T_final <= min(T_i)` (bounded relaxation of 1 tier in ACWCM mode).
- **Abstention Correctness (AC):** Any gate returning `R*` forces the system output to `R*`.
- **Critical Non-Dilution (CND):** `R1` or `R2` from any gate must propagate to the final tier (capped at `R2`).

### Testing

Run the full test suite before submitting:

```bash
python tests/test_full_system.py
```

All 10 tests must pass. If you add a new gate or baseline, add corresponding
tests in `tests/`.

### Clinical Thresholds

All clinical thresholds (e.g. GCS < 14, SpO2 < 92%) must cite published
guidelines (AHA/ASA, ACEP, or peer-reviewed literature). Do not change
thresholds without a supporting reference.

## Pull Request Process

1. Ensure the test suite passes with zero failures.
2. Update documentation if the public interface changes.
3. Write a clear PR description summarising what and why.
4. One approval is required before merging.

## Reporting Issues

Use [GitHub Issues](https://github.com/ChatchaiTritham/SAFE-Gate/issues).
Include:

- Python version and operating system.
- Minimal reproducible example.
- Expected vs. actual behaviour.

## Code of Conduct

Be respectful and constructive. This project deals with patient safety;
accuracy and rigour are more important than speed.
