#!/bin/bash
cd "$(dirname "$0")"
./venv/bin/python web_app.py "$@"
