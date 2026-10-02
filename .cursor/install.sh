#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
fi
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
