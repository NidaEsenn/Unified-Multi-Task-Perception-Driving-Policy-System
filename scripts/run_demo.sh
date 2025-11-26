#!/usr/bin/env bash
set -euo pipefail

# Small helper to run the overlay demo with default arguments.
python -m realtime_demo.overlay_demo --source 0 "$@"
