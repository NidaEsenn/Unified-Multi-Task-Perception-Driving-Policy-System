#!/usr/bin/env bash
set -euo pipefail

echo "Running smoke test..."
python -m tests.smoke_test
echo "Smoke test finished."
