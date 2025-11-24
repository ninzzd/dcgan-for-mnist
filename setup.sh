#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

echo "==> Using Python: $(command -v python3 || echo 'python3 not found')"

# 1) Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "==> Virtual environment already exists at $VENV_DIR"
fi

# 2) Activate venv
# shellcheck source=/dev/null
echo "==> Activating virtual environment"
source "$VENV_DIR/bin/activate"

# 3) Upgrade pip
echo "==> Upgrading pip"
python -m pip install --upgrade pip

# 4) Install requirements
if [ -f "requirements.txt" ]; then
    echo "==> Installing dependencies from requirements.txt"
    python -m pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found; skipping dependency install."
fi

echo
echo "Setup complete."
echo "To activate the environment later, run:"
echo "  source $VENV_DIR/bin/activate"
