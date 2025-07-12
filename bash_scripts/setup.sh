#!/usr/bin/env bash

# Fail on any error
set -e

echo "🌟 Starting setup..."

echo "🚀 Installing Python requirements (from inside setup script)..."
python setup.py

echo "🎉 Setup complete! ✅"