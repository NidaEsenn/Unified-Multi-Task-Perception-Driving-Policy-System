# Makefile for common developer tasks

PYTHON := python
PIP := $(PYTHON) -m pip

.PHONY: install install-dev smoke-test test run-demo clean

install:
    $(PIP) install --upgrade pip
    $(PIP) install -r requirements.txt

install-dev:
    $(PIP) install --upgrade pip
    # Use CPU-only PyTorch wheel for CI/dev to avoid CUDA requirements
    $(PIP) install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    $(PIP) install -r requirements-dev.txt

smoke-test:
    $(PYTHON) -m tests.smoke_test

test:
    pytest -q

run-demo:
    $(PYTHON) -m realtime_demo.overlay_demo --source 0

clean:
    find . -type d -name '__pycache__' -exec rm -rf {} + || true
    rm -rf build dist *.egg-info || true
