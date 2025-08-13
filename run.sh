#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/fetch_and_ttest.py

echo
echo "Hotovo. Výsledek je ve složkách plots/ (PNG) a results/ (CSV + results_summary.md)"
