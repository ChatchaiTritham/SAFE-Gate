#!/bin/bash
# SAFE-Gate Modern Tools Setup Script
# Quick setup for essential development tools

set -e  # Exit on error

echo "========================================================================"
echo "SAFE-Gate Modern Tools Setup"
echo "========================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python $python_version detected (>= $required_version)"
else
    print_error "Python $required_version or higher required (found $python_version)"
    exit 1
fi

# Install development dependencies
print_info "Installing development dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Install code quality tools
print_info "Installing code quality tools..."
pip install -q black pylint mypy autopep8 isort flake8

# Install testing tools
print_info "Installing testing tools..."
pip install -q pytest pytest-cov pytest-benchmark pytest-mock

# Install security tools
print_info "Installing security tools..."
pip install -q safety bandit

# Install pre-commit
print_info "Installing pre-commit..."
pip install -q pre-commit

print_success "All dependencies installed"

# Create directory structure
print_info "Creating directory structure..."
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p benchmarks
mkdir -p logs
mkdir -p .github/workflows

print_success "Directory structure created"

# Setup pre-commit hooks
print_info "Setting up pre-commit hooks..."

cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.9
        args: ['--line-length=100']

  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--profile', 'black', '--line-length', '100']

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,E501']
EOF

pre-commit install
print_success "Pre-commit hooks configured"

# Create pytest configuration
print_info "Creating pytest configuration..."

cat > pytest.ini << 'EOF'
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
    --maxfail=3
    -v

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance/benchmark tests
    slow: Tests that take significant time
EOF

print_success "Pytest configuration created"

# Create basic test file
print_info "Creating basic test file..."

cat > tests/unit/test_imports.py << 'EOF'
"""Basic import tests for SAFE-Gate components."""

def test_safegate_import():
    """Test SAFE-Gate main class import."""
    from src.safegate import SAFEGate
    assert SAFEGate is not None

def test_risk_lattice_import():
    """Test risk lattice imports."""
    from src.merging.risk_lattice import RiskTier, RiskLattice
    assert RiskTier is not None
    assert RiskLattice is not None

def test_conservative_merging_import():
    """Test conservative merging import."""
    from src.merging.conservative_merging import ConservativeMerging
    assert ConservativeMerging is not None

def test_gates_import():
    """Test all gates import."""
    from src.gates.gate1_critical_flags import Gate1CriticalFlags
    from src.gates.gate2_moderate_risk import Gate2ModerateRisk
    from src.gates.gate3_data_quality import Gate3DataQuality
    from src.gates.gate4_titrate_logic import Gate4TiTrATELogic
    from src.gates.gate5_uncertainty import Gate5Uncertainty
    from src.gates.gate6_temporal_risk import Gate6TemporalRisk

    assert all([
        Gate1CriticalFlags, Gate2ModerateRisk, Gate3DataQuality,
        Gate4TiTrATELogic, Gate5Uncertainty, Gate6TemporalRisk
    ])
EOF

print_success "Basic test file created"

# Create GitHub Actions workflow
print_info "Creating GitHub Actions CI workflow..."

cat > .github/workflows/ci.yml << 'EOF'
name: SAFE-Gate CI

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
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black pylint safety

    - name: Code formatting check
      run: black --check src/ tests/ evaluation/

    - name: Run security check
      run: safety check --ignore 70612

    - name: Run tests
      run: pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
EOF

print_success "GitHub Actions workflow created"

# Create .gitignore additions
print_info "Updating .gitignore..."

cat >> .gitignore << 'EOF'

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
*.cover

# Code quality
.mypy_cache/
.pylint.d/
.black.d/

# Logs
logs/
*.log

# Benchmarks
benchmarks/results/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

print_success ".gitignore updated"

# Run initial checks
print_info "Running initial code quality checks..."

echo ""
echo "→ Running Black (code formatter)..."
black --check src/ tests/ evaluation/ 2>/dev/null && print_success "Code formatting: PASS" || {
    echo "  Fixing formatting..."
    black src/ tests/ evaluation/
    print_success "Code formatted"
}

echo ""
echo "→ Running pytest (tests)..."
pytest tests/ -v --maxfail=1 && print_success "Tests: PASS" || print_error "Tests: FAILED (check output above)"

echo ""
echo "→ Running safety (security check)..."
safety check --ignore 70612 2>/dev/null && print_success "Security check: PASS" || print_error "Security check: WARNING (check vulnerabilities)"

echo ""
echo "========================================================================"
print_success "SAFE-Gate modern tools setup complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Review .pre-commit-config.yaml and adjust as needed"
echo "  2. Run: pre-commit run --all-files"
echo "  3. Run: pytest tests/ --cov=src"
echo "  4. Push changes to trigger GitHub Actions"
echo ""
echo "Available commands:"
echo "  black src/              - Format code"
echo "  pytest tests/           - Run tests"
echo "  pytest --cov=src        - Run tests with coverage"
echo "  safety check            - Security scan"
echo "  pre-commit run --all    - Run all pre-commit hooks"
echo ""
echo "Documentation:"
echo "  See MODERNIZATION_ROADMAP.md for complete guide"
echo ""
