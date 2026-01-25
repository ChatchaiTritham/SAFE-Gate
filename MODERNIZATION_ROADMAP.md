# SAFE-Gate Modernization Roadmap

**Version:** 1.0
**Date:** 2026-01-25
**Purpose:** Strategic plan for modernizing SAFE-Gate repository with industry best practices

---

## Executive Summary

This document outlines modern software engineering practices and tools to enhance the SAFE-Gate repository's quality, maintainability, reproducibility, and deployment readiness. All recommendations follow current industry standards and are widely adopted in production ML systems.

**Key Benefits:**
- ✅ Automated quality assurance
- ✅ Reproducible experiments
- ✅ Easy deployment
- ✅ Better collaboration
- ✅ Enhanced documentation
- ✅ Continuous improvement

---

## 1. Continuous Integration/Continuous Deployment (CI/CD)

### Current State
- Manual testing
- No automated checks
- Manual deployment process

### Proposed Solution: GitHub Actions

**Implementation:**

Create `.github/workflows/ci.yml`:

```yaml
name: SAFE-Gate CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black pylint mypy

    - name: Code formatting check (Black)
      run: black --check src/ tests/

    - name: Linting (Pylint)
      run: pylint src/

    - name: Type checking (MyPy)
      run: mypy src/

    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build documentation
      run: |
        pip install sphinx sphinx-rtd-theme
        cd docs && make html
```

**Benefits:**
- ✅ Automatic testing on every commit
- ✅ Multi-version Python compatibility
- ✅ Code quality enforcement
- ✅ Coverage tracking
- ✅ Documentation validation

**Priority:** HIGH
**Effort:** Medium (2-3 hours)
**Impact:** High

---

## 2. Automated Testing Framework

### Current State
- Basic test scripts
- No test coverage metrics
- No regression testing

### Proposed Solution: Pytest + Coverage

**Structure:**
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_gate1_critical.py
│   ├── test_gate2_moderate.py
│   ├── test_gate3_quality.py
│   ├── test_gate4_titrate.py
│   ├── test_gate5_uncertainty.py
│   ├── test_gate6_temporal.py
│   ├── test_conservative_merging.py
│   └── test_risk_lattice.py
├── integration/
│   ├── test_safegate_system.py
│   └── test_full_pipeline.py
└── performance/
    ├── test_latency.py
    └── test_throughput.py
```

**Example Test:**

```python
# tests/unit/test_gate1_critical.py
import pytest
import pandas as pd
from src.gates.gate1_critical_flags import Gate1CriticalFlags
from src.merging.risk_lattice import RiskTier

@pytest.fixture
def gate1():
    """Fixture for Gate 1 instance."""
    return Gate1CriticalFlags()

@pytest.fixture
def critical_patient():
    """Fixture for critical patient data."""
    return pd.DataFrame([{
        'systolic_bp': 85,
        'heart_rate': 125,
        'focal_neuro_deficit': True,
        'altered_mental_status': True,
        'gcs': 13,
        # ... other fields
    }])

@pytest.fixture
def safe_patient():
    """Fixture for safe patient data."""
    return pd.DataFrame([{
        'systolic_bp': 120,
        'heart_rate': 72,
        'focal_neuro_deficit': False,
        'altered_mental_status': False,
        'gcs': 15,
        # ... other fields
    }])

def test_gate1_detects_critical_case(gate1, critical_patient):
    """Test Gate 1 correctly identifies critical case."""
    result = gate1.evaluate(critical_patient)
    assert result[0] in [RiskTier.R_STAR, RiskTier.R1, RiskTier.R2]

def test_gate1_allows_safe_case(gate1, safe_patient):
    """Test Gate 1 doesn't over-escalate safe cases."""
    result = gate1.evaluate(safe_patient)
    assert result[0] != RiskTier.R1

def test_gate1_handles_missing_data(gate1):
    """Test Gate 1 handles missing data gracefully."""
    incomplete_data = pd.DataFrame([{'systolic_bp': 120}])
    result = gate1.evaluate(incomplete_data)
    assert result[0] == RiskTier.R_STAR  # Should abstain

@pytest.mark.performance
def test_gate1_latency(gate1, critical_patient, benchmark):
    """Test Gate 1 meets latency requirements (<2ms)."""
    result = benchmark(gate1.evaluate, critical_patient)
    # Benchmark automatically measures execution time
```

**Configuration: `pytest.ini`**

```ini
[pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -ra
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --maxfail=1
    -v

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance/benchmark tests
    slow: Tests that take significant time
```

**Benefits:**
- ✅ 80%+ code coverage
- ✅ Automated regression detection
- ✅ Performance benchmarking
- ✅ Clear test organization

**Priority:** HIGH
**Effort:** High (1-2 days)
**Impact:** Very High

---

## 3. Code Quality Tools

### a. Black (Code Formatter)

**Installation:**
```bash
pip install black
```

**Configuration: `pyproject.toml`**

```toml
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | build
  | dist
)/
'''
```

**Usage:**
```bash
# Format all code
black src/ tests/ evaluation/

# Check without modifying
black --check src/

# Show diff
black --diff src/
```

---

### b. Pylint (Linter)

**Configuration: `.pylintrc`**

```ini
[MASTER]
ignore=CVS,.git,__pycache__
jobs=4

[MESSAGES CONTROL]
disable=
    C0330,  # Wrong hanging indentation
    C0103,  # Invalid name (allows lowercase variables)
    R0913,  # Too many arguments
    R0914,  # Too many local variables
    W0212,  # Protected access (allows _private)

[FORMAT]
max-line-length=100
indent-string='    '

[DESIGN]
max-args=8
max-attributes=12
max-public-methods=25

[SIMILARITIES]
min-similarity-lines=4
```

**Usage:**
```bash
pylint src/ --output-format=colorized
pylint src/ --output-format=json > pylint-report.json
```

---

### c. MyPy (Type Checker)

**Configuration: `mypy.ini`**

```ini
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = True
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-numpy.*]
ignore_missing_imports = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True
```

**Usage:**
```bash
mypy src/
mypy --strict src/gates/
```

---

### d. Pre-commit Hooks

**Configuration: `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: check-json
      - id: check-toml
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/PyCQA/pylint
    rev: v3.0.3
    hooks:
      - id: pylint
        args: ['--disable=C0330,C0103']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Setup:**
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files  # Test all hooks
```

**Benefits:**
- ✅ Consistent code style
- ✅ Catch bugs before commit
- ✅ Type safety
- ✅ Automated quality checks

**Priority:** MEDIUM
**Effort:** Low (1 hour)
**Impact:** High

---

## 4. Documentation System

### Proposed Solution: Sphinx + Read the Docs

**Structure:**
```
docs/
├── source/
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── api/
│   │   ├── gates.rst
│   │   ├── merging.rst
│   │   └── safegate.rst
│   ├── tutorials/
│   │   ├── basic_usage.rst
│   │   └── advanced.rst
│   ├── performance/
│   │   └── benchmarks.rst
│   └── contributing.rst
├── Makefile
└── make.bat
```

**Configuration: `docs/source/conf.py`**

```python
project = 'SAFE-Gate'
copyright = '2026, Chatchai Tritham, Chakkrit Snae Namahoot'
author = 'Chatchai Tritham, Chakkrit Snae Namahoot'
version = '1.0'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy docstring support
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
```

**Example Docstring:**

```python
class Gate1CriticalFlags:
    """
    Gate 1: Critical Flags Detection.

    Detects life-threatening red flags through rule-based logic for
    hemodynamic instability, altered consciousness, and acute focal
    neurological deficits.

    Attributes:
        thresholds (dict): Critical vital sign thresholds.
        severe_thresholds (dict): Severe single-flag R1 triggers.
        neurological_flags (set): Central stroke signs.

    Examples:
        >>> from safegate import SAFEGate
        >>> gate1 = Gate1CriticalFlags()
        >>> patient_data = pd.DataFrame([{
        ...     'systolic_bp': 85,
        ...     'heart_rate': 125,
        ...     'gcs': 13
        ... }])
        >>> result = gate1.evaluate(patient_data)
        >>> print(result[0])
        RiskTier.R1

    References:
        Tritham, C., & Snae Namahoot, C. (2026). SAFE-Gate: A Knowledge-Based
        Expert System for Emergency Triage Safety. Expert Systems with
        Applications.
    """

    def evaluate(self, patient_data: pd.DataFrame) -> List[RiskTier]:
        """
        Evaluate patient risk using critical flags logic.

        Args:
            patient_data (pd.DataFrame): Patient features with required columns:
                - systolic_bp: Systolic blood pressure (mmHg)
                - heart_rate: Heart rate (bpm)
                - gcs: Glasgow Coma Scale (3-15)
                - focal_neuro_deficit: Boolean flag
                - altered_mental_status: Boolean flag

        Returns:
            List[RiskTier]: Risk tier assessment for each patient.
                - R1: Critical (life-threatening)
                - R2-R5: Lower risk tiers
                - R_STAR: Abstention (missing data)

        Raises:
            ValueError: If required columns are missing.
            TypeError: If patient_data is not a DataFrame.

        Examples:
            >>> gate = Gate1CriticalFlags()
            >>> df = pd.DataFrame([{'systolic_bp': 85, 'heart_rate': 125}])
            >>> result = gate.evaluate(df)
            >>> assert result[0] == RiskTier.R1

        Notes:
            This gate implements deterministic rules without ML components,
            ensuring verifiable and stable performance.
        """
        pass
```

**Build Documentation:**

```bash
cd docs
make html              # Build HTML
make latexpdf          # Build PDF
make clean && make html  # Clean rebuild
```

**Host on Read the Docs:**
- Free hosting at `safegate.readthedocs.io`
- Automatic builds on Git commits
- Version support (v1.0, latest, stable)

**Benefits:**
- ✅ Professional API documentation
- ✅ Searchable documentation
- ✅ Auto-generated from docstrings
- ✅ Version control

**Priority:** MEDIUM
**Effort:** Medium (4-6 hours)
**Impact:** High

---

## 5. Docker Containerization

### Proposed Solution: Multi-stage Docker Build

**Dockerfile:**

```dockerfile
# SAFE-Gate Docker Image
# Multi-stage build for optimized production deployment

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 safegate && \
    mkdir -p /app && \
    chown -R safegate:safegate /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/safegate/.local
ENV PATH=/home/safegate/.local/bin:$PATH

# Copy application code
COPY --chown=safegate:safegate src/ ./src/
COPY --chown=safegate:safegate evaluation/ ./evaluation/
COPY --chown=safegate:safegate *.py ./

# Switch to non-root user
USER safegate

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from src.safegate import SAFEGate; SAFEGate()" || exit 1

# Default command
CMD ["python", "-m", "src.safegate"]

# Metadata
LABEL maintainer="Chatchai Tritham <chatchait66@nu.ac.th>"
LABEL description="SAFE-Gate: Expert System for Emergency Triage"
LABEL version="1.0.0"
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  safegate:
    build:
      context: .
      dockerfile: Dockerfile
    image: safegate:latest
    container_name: safegate-system
    restart: unless-stopped
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
      - ./models:/app/models:ro
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    ports:
      - "8000:8000"  # API port
    healthcheck:
      test: ["CMD", "python", "-c", "from src.safegate import SAFEGate; SAFEGate()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add FastAPI service
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: safegate-api:latest
    container_name: safegate-api
    restart: unless-stopped
    depends_on:
      - safegate
    ports:
      - "8080:8080"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8080
```

**Usage:**

```bash
# Build image
docker build -t safegate:latest .

# Run container
docker run -d --name safegate -p 8000:8000 safegate:latest

# With docker-compose
docker-compose up -d

# View logs
docker logs -f safegate

# Execute commands inside container
docker exec -it safegate python evaluation/create_manuscript_figures.py

# Stop and remove
docker-compose down
```

**.dockerignore:**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.git/
.github/
.vscode/
.idea/
*.egg-info/
dist/
build/
*.log
.pytest_cache/
.coverage
htmlcov/
.DS_Store
*.swp
*.swo
```

**Benefits:**
- ✅ Reproducible environment
- ✅ Easy deployment
- ✅ Isolated dependencies
- ✅ Scalable

**Priority:** MEDIUM
**Effort:** Low (2 hours)
**Impact:** High

---

## 6. REST API with FastAPI

### Proposed Solution: FastAPI + Pydantic

**Structure:**
```
api/
├── __init__.py
├── main.py              # FastAPI app
├── models.py            # Pydantic models
├── endpoints/
│   ├── __init__.py
│   ├── predict.py
│   └── health.py
└── tests/
    └── test_api.py
```

**api/main.py:**

```python
"""
SAFE-Gate REST API
FastAPI implementation for clinical deployment
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import uvicorn
from datetime import datetime

from src.safegate import SAFEGate
from src.merging.risk_lattice import RiskTier

# Initialize FastAPI app
app = FastAPI(
    title="SAFE-Gate API",
    description="REST API for SAFE-Gate Emergency Triage Expert System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PatientInput(BaseModel):
    """Patient data input schema."""

    # Demographics
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., regex="^(M|F|Other)$", description="Patient gender")

    # Vital signs
    systolic_bp: int = Field(..., ge=50, le=250, description="Systolic BP (mmHg)")
    diastolic_bp: int = Field(..., ge=30, le=150, description="Diastolic BP (mmHg)")
    heart_rate: int = Field(..., ge=30, le=200, description="Heart rate (bpm)")
    respiratory_rate: int = Field(..., ge=5, le=50, description="Respiratory rate")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature (°C)")
    spo2: int = Field(..., ge=70, le=100, description="SpO2 (%)")

    # Neurological
    gcs: int = Field(..., ge=3, le=15, description="Glasgow Coma Scale")
    focal_neuro_deficit: bool = Field(..., description="Focal neurological deficit")
    altered_mental_status: bool = Field(..., description="Altered mental status")

    # Optional clinical features
    hypertension: Optional[bool] = None
    diabetes: Optional[bool] = None
    atrial_fibrillation: Optional[bool] = None

    @validator('diastolic_bp')
    def validate_bp_ratio(cls, v, values):
        """Ensure diastolic BP < systolic BP."""
        if 'systolic_bp' in values and v >= values['systolic_bp']:
            raise ValueError('Diastolic BP must be less than systolic BP')
        return v

    class Config:
        schema_extra = {
            "example": {
                "age": 75,
                "gender": "M",
                "systolic_bp": 85,
                "diastolic_bp": 50,
                "heart_rate": 125,
                "respiratory_rate": 22,
                "temperature": 37.5,
                "spo2": 92,
                "gcs": 13,
                "focal_neuro_deficit": True,
                "altered_mental_status": True,
                "hypertension": True,
                "diabetes": True,
                "atrial_fibrillation": False
            }
        }

class PredictionResponse(BaseModel):
    """API response schema."""

    patient_id: str
    risk_tier: str
    risk_description: str
    enforcing_gate: str
    confidence: float
    gate_outputs: dict
    timestamp: datetime
    latency_ms: float

    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "risk_tier": "R1",
                "risk_description": "Critical: Life-threatening, immediate care required",
                "enforcing_gate": "G1",
                "confidence": 0.95,
                "gate_outputs": {
                    "G1": "R1",
                    "G2": "R2",
                    "G3": "R3",
                    "G4": "R2",
                    "G5": "R3",
                    "G6": "R2"
                },
                "timestamp": "2026-01-25T14:30:00",
                "latency_ms": 1.5
            }
        }

# Global SAFE-Gate instance
safegate_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize SAFE-Gate on startup."""
    global safegate_system
    safegate_system = SAFEGate()
    print("✓ SAFE-Gate system initialized")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAFE-Gate API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "system": "SAFE-Gate v1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    patient: PatientInput,
    patient_id: Optional[str] = None
):
    """
    Classify patient risk tier.

    Args:
        patient: Patient data
        patient_id: Optional patient identifier

    Returns:
        PredictionResponse with risk assessment

    Raises:
        HTTPException: If prediction fails
    """
    try:
        import pandas as pd
        import time

        # Convert to DataFrame
        patient_dict = patient.dict(exclude_none=True)
        df = pd.DataFrame([patient_dict])

        # Make prediction
        start_time = time.time()
        result = safegate_system.classify(
            patient_dict,
            patient_id=patient_id or f"API_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        latency_ms = (time.time() - start_time) * 1000

        # Format response
        return PredictionResponse(
            patient_id=result.get('patient_id', patient_id),
            risk_tier=result['final_tier'],
            risk_description=RiskTier[result['final_tier']].get_description(),
            enforcing_gate=result['enforcing_gate'],
            confidence=result['confidence'],
            gate_outputs=result.get('gate_outputs', {}),
            timestamp=datetime.now(),
            latency_ms=latency_ms
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(patients: List[PatientInput]):
    """Batch prediction endpoint."""
    results = []
    for i, patient in enumerate(patients):
        try:
            result = await predict(patient, patient_id=f"BATCH_{i}")
            results.append(result)
        except Exception as e:
            # Log error but continue processing
            print(f"Error processing patient {i}: {e}")
    return results

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

**Run API:**

```bash
# Install FastAPI and Uvicorn
pip install fastapi uvicorn[standard] pydantic

# Run development server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production with multiple workers
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Test API:**

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 75,
    "gender": "M",
    "systolic_bp": 85,
    "diastolic_bp": 50,
    "heart_rate": 125,
    "respiratory_rate": 22,
    "temperature": 37.5,
    "spo2": 92,
    "gcs": 13,
    "focal_neuro_deficit": true,
    "altered_mental_status": true
  }'

# Interactive docs
# Open browser: http://localhost:8000/docs
```

**Benefits:**
- ✅ RESTful API for integration
- ✅ Auto-generated OpenAPI docs
- ✅ Input validation
- ✅ Production-ready

**Priority:** LOW (for research) / HIGH (for deployment)
**Effort:** Medium (3-4 hours)
**Impact:** Very High (for clinical deployment)

---

## 7. Model Versioning & Experiment Tracking

### Proposed Solution: DVC + MLflow

**a. DVC (Data Version Control)**

**Setup:**

```bash
pip install dvc dvc-gdrive

# Initialize DVC
dvc init

# Add remote storage (Google Drive example)
dvc remote add -d storage gdrive://1ABC...XYZ

# Track data and models
dvc add data/raw/patient_data.csv
dvc add models/gate2_xgboost.pkl

# Push to remote
dvc push
```

**dvc.yaml (Pipeline definition):**

```yaml
stages:
  prepare:
    cmd: python src/prepare_data.py
    deps:
      - data/raw/patient_data.csv
      - src/prepare_data.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    params:
      - prepare.train_split
      - prepare.random_seed

  train:
    cmd: python src/train.py
    deps:
      - data/processed/train.csv
      - src/train.py
    outs:
      - models/gate2_xgboost.pkl
    params:
      - train.max_depth
      - train.learning_rate
      - train.n_estimators
    metrics:
      - metrics/train.json:
          cache: false

  evaluate:
    cmd: python evaluation/generate_performance_metrics.py
    deps:
      - models/gate2_xgboost.pkl
      - data/processed/test.csv
      - evaluation/generate_performance_metrics.py
    metrics:
      - metrics/evaluation.json:
          cache: false
    plots:
      - plots/confusion_matrix.png
      - plots/per_class_metrics.png
```

**params.yaml:**

```yaml
prepare:
  train_split: 0.8
  random_seed: 42

train:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
  subsample: 0.8
  colsample_bytree: 0.8
```

---

**b. MLflow (Experiment Tracking)**

**Integration:**

```python
# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

# Set experiment
mlflow.set_experiment("SAFE-Gate-Training")

# Start run
with mlflow.start_run(run_name="gate2_xgboost_v1"):
    # Log parameters
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("n_estimators", 100)

    # Train model
    model = train_xgboost(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("plots/confusion_matrix.png")
    mlflow.log_artifact("metrics/evaluation.json")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

**Run MLflow UI:**

```bash
mlflow ui --port 5000
# Open browser: http://localhost:5000
```

**Benefits:**
- ✅ Track experiments
- ✅ Version datasets and models
- ✅ Reproducible pipelines
- ✅ Compare runs

**Priority:** MEDIUM
**Effort:** Medium (4-6 hours)
**Impact:** High (for research reproducibility)

---

## 8. Performance Benchmarking

### Proposed Solution: Automated Benchmarks

**benchmarks/benchmark_suite.py:**

```python
"""
SAFE-Gate Performance Benchmark Suite
"""

import time
import psutil
import pandas as pd
import numpy as np
from memory_profiler import profile
from contextlib import contextmanager

@contextmanager
def measure_time():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000  # ms

@contextmanager
def measure_memory():
    """Context manager for measuring memory usage."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    yield lambda: (process.memory_info().rss / 1024 / 1024) - mem_before

class SAFEGateBenchmark:
    """Benchmark suite for SAFE-Gate system."""

    def __init__(self):
        from src.safegate import SAFEGate
        self.safegate = SAFEGate()
        self.results = []

    def benchmark_single_prediction(self, n_runs=1000):
        """Benchmark single patient predictions."""
        patient = self._generate_patient()

        latencies = []
        for _ in range(n_runs):
            with measure_time() as get_time:
                self.safegate.classify(patient)
            latencies.append(get_time())

        return {
            'test': 'single_prediction',
            'n_runs': n_runs,
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
        }

    def benchmark_throughput(self, duration_seconds=10):
        """Benchmark system throughput."""
        start_time = time.time()
        count = 0

        while time.time() - start_time < duration_seconds:
            patient = self._generate_patient()
            self.safegate.classify(patient)
            count += 1

        elapsed = time.time() - start_time
        throughput = count / elapsed

        return {
            'test': 'throughput',
            'duration_seconds': duration_seconds,
            'total_predictions': count,
            'throughput_per_second': throughput,
        }

    @profile
    def benchmark_memory(self):
        """Profile memory usage."""
        patients = [self._generate_patient() for _ in range(100)]

        with measure_memory() as get_memory:
            for patient in patients:
                self.safegate.classify(patient)

        return {
            'test': 'memory_usage',
            'memory_increase_mb': get_memory(),
        }

    def run_all_benchmarks(self):
        """Run complete benchmark suite."""
        print("Running SAFE-Gate Benchmark Suite...")

        # Single prediction benchmark
        print("\n[1/3] Single Prediction Latency...")
        result1 = self.benchmark_single_prediction(n_runs=1000)
        self.results.append(result1)
        print(f"  Mean: {result1['mean_latency_ms']:.2f}ms")
        print(f"  P95: {result1['p95_latency_ms']:.2f}ms")
        print(f"  P99: {result1['p99_latency_ms']:.2f}ms")

        # Throughput benchmark
        print("\n[2/3] Throughput...")
        result2 = self.benchmark_throughput(duration_seconds=10)
        self.results.append(result2)
        print(f"  Throughput: {result2['throughput_per_second']:.2f} predictions/sec")

        # Memory benchmark
        print("\n[3/3] Memory Usage...")
        result3 = self.benchmark_memory()
        self.results.append(result3)
        print(f"  Memory: {result3['memory_increase_mb']:.2f} MB")

        return pd.DataFrame(self.results)

    @staticmethod
    def _generate_patient():
        """Generate random patient data."""
        return {
            'age': np.random.randint(20, 90),
            'gender': np.random.choice(['M', 'F']),
            'systolic_bp': np.random.randint(80, 180),
            'diastolic_bp': np.random.randint(50, 110),
            'heart_rate': np.random.randint(50, 150),
            # ... other features
        }

if __name__ == "__main__":
    benchmark = SAFEGateBenchmark()
    results_df = benchmark.run_all_benchmarks()

    # Save results
    results_df.to_csv('benchmarks/results.csv', index=False)
    print("\n✓ Benchmark results saved to benchmarks/results.csv")
```

**Run Benchmarks:**

```bash
python benchmarks/benchmark_suite.py
```

**Benefits:**
- ✅ Track performance over time
- ✅ Identify regressions
- ✅ Optimize bottlenecks
- ✅ Validate latency requirements (<2ms)

**Priority:** MEDIUM
**Effort:** Low (2 hours)
**Impact:** Medium

---

## 9. Security & Dependency Management

### a. Dependabot (Automated Dependency Updates)

**Configuration: `.github/dependabot.yml`**

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "ChatchaiTritham"
    labels:
      - "dependencies"
      - "python"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "github-actions"
```

---

### b. Safety (Security Scanner)

**Usage:**

```bash
# Install
pip install safety

# Check for known vulnerabilities
safety check

# Check with full report
safety check --full-report

# Check specific requirements file
safety check -r requirements.txt

# Generate JSON report
safety check --json > security-report.json
```

---

### c. Bandit (Security Linter)

**Configuration: `.bandit`**

```yaml
exclude_dirs:
  - /test
  - /tests
  - /venv

tests:
  - B201  # Flask debug mode
  - B301  # Pickle usage
  - B302  # Insecure SSL
  - B303  # MD5 usage
  - B304  # Insecure cipher
  - B305  # Insecure cipher mode
  - B306  # Insecure mktemp
  - B307  # Eval usage
```

**Usage:**

```bash
# Install
pip install bandit

# Run security scan
bandit -r src/

# Generate HTML report
bandit -r src/ -f html -o bandit-report.html
```

**Benefits:**
- ✅ Automated security updates
- ✅ Vulnerability scanning
- ✅ Code security analysis
- ✅ Supply chain security

**Priority:** HIGH
**Effort:** Low (1 hour)
**Impact:** High

---

## 10. Monitoring & Logging

### Proposed Solution: Structured Logging

**utils/logging_config.py:**

```python
"""
Structured logging configuration for SAFE-Gate
"""

import logging
import logging.config
import json
from datetime import datetime
from pathlib import Path

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        },
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/safegate.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json',
            'filename': 'logs/safegate_errors.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'loggers': {
        'safegate': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

def setup_logging():
    """Initialize logging configuration."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(LOGGING_CONFIG)

    logger = logging.getLogger('safegate')
    logger.info("Logging initialized", extra={
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

    return logger

# Usage
logger = setup_logging()
```

**Integration:**

```python
# src/safegate.py
from utils.logging_config import logger

class SAFEGate:
    def classify(self, patient_data, patient_id=None):
        logger.info("Classification started", extra={
            'patient_id': patient_id,
            'features': len(patient_data)
        })

        try:
            # ... classification logic ...

            logger.info("Classification completed", extra={
                'patient_id': patient_id,
                'risk_tier': result['final_tier'],
                'enforcing_gate': result['enforcing_gate'],
                'latency_ms': latency
            })

            return result

        except Exception as e:
            logger.error("Classification failed", extra={
                'patient_id': patient_id,
                'error': str(e)
            }, exc_info=True)
            raise
```

**Benefits:**
- ✅ Structured JSON logs
- ✅ Log rotation
- ✅ Debug tracking
- ✅ Error monitoring

**Priority:** MEDIUM
**Effort:** Low (1 hour)
**Impact:** High (for production)

---

## Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Timeline |
|---------|----------|--------|--------|----------|
| **1. CI/CD Pipeline** | HIGH | Medium | High | Week 1 |
| **2. Automated Testing** | HIGH | High | Very High | Week 1-2 |
| **3. Code Quality Tools** | MEDIUM | Low | High | Week 1 |
| **4. Documentation** | MEDIUM | Medium | High | Week 2 |
| **5. Docker** | MEDIUM | Low | High | Week 1 |
| **6. REST API** | LOW/HIGH* | Medium | Very High* | Week 3 |
| **7. Model Versioning** | MEDIUM | Medium | High | Week 2 |
| **8. Benchmarking** | MEDIUM | Low | Medium | Week 2 |
| **9. Security** | HIGH | Low | High | Week 1 |
| **10. Logging** | MEDIUM | Low | High | Week 1 |

*Priority depends on deployment needs (LOW for research, HIGH for clinical deployment)

---

## Phase 1 (Week 1): Foundation
- ✅ CI/CD with GitHub Actions
- ✅ Pre-commit hooks
- ✅ Docker containerization
- ✅ Security scanning
- ✅ Basic logging

## Phase 2 (Week 2): Quality & Documentation
- ✅ Comprehensive test suite
- ✅ Sphinx documentation
- ✅ DVC setup
- ✅ Performance benchmarks

## Phase 3 (Week 3): Advanced Features
- ✅ FastAPI REST API
- ✅ MLflow integration
- ✅ Advanced monitoring
- ✅ Production deployment guide

---

## Quick Start Implementation

**Day 1: Essential Setup (2-3 hours)**

```bash
# 1. Install tools
pip install black pylint mypy pre-commit pytest safety

# 2. Setup pre-commit
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/pylint
    rev: v3.0.3
    hooks:
      - id: pylint
EOF

pre-commit install
pre-commit run --all-files

# 3. Create basic tests
mkdir -p tests/unit
cat > tests/unit/test_basic.py << 'EOF'
def test_import():
    from src.safegate import SAFEGate
    assert SAFEGate is not None
EOF

# 4. Run security check
safety check

# 5. Format code
black src/ tests/ evaluation/

echo "✓ Basic modernization complete!"
```

---

## Conclusion

This modernization roadmap provides a systematic approach to enhancing the SAFE-Gate repository with industry-standard practices. Implementation can be phased based on priorities and resources.

**Expected Outcomes:**
- 📈 **Code Quality:** Consistent, maintainable, type-safe code
- 🧪 **Testing:** 80%+ coverage with automated regression detection
- 🚀 **Deployment:** Easy deployment with Docker and API
- 📚 **Documentation:** Professional, searchable documentation
- 🔒 **Security:** Continuous security monitoring
- 🔄 **Reproducibility:** Version-controlled data and experiments

**Maintenance:**
- All tools integrate into development workflow
- Automated checks prevent quality degradation
- CI/CD ensures continuous validation
- Documentation stays synchronized with code

---

**Next Steps:**
1. Review and prioritize features
2. Begin Phase 1 implementation
3. Iterate based on feedback
4. Expand to Phase 2 and 3

**Contact:**
- Chatchai Tritham: chatchait66@nu.ac.th
- GitHub: https://github.com/ChatchaiTritham/SAFE-Gate

---

**Document Version:** 1.0
**Last Updated:** 2026-01-25
