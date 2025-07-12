#!/usr/bin/env bash

# Exit if any command fails
set -e

# -------------------------------
# Configuration
# -------------------------------
CONFIG_PATH="aml/config/default.yaml"

# -------------------------------
# Parse arguments
# -------------------------------
MOCK_FLAG=""
if [ "$1" == "--mock" ]; then
  MOCK_FLAG="--mock"
  echo "⚡ Running in mock mode (sanity check)."
fi

# -------------------------------
# Run the Python pipeline
# -------------------------------
echo "🚀 Starting SignCLIP evaluation pipeline..."
python3 aml/main.py --config "$CONFIG_PATH" $MOCK_FLAG

echo "✅ Finished SignCLIP evaluation pipeline."
