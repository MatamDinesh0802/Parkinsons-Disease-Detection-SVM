#!/usr/bin/env bash
# One-shot setup: creates a project-local .venv and installs dependencies.
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment at .venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
python -m pip install --upgrade pip --quiet

echo "Installing requirements ..."
pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate with:  source .venv/bin/activate"
echo "Train model:    python -m src.parkinsons.train"
echo "Run demo:       streamlit run app/streamlit_app.py"
