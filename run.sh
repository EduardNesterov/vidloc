#!/bin/sh
# Launch the Vidloc GUI (local video localizer).
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DIR"
exec "$DIR/.venv/bin/python" -m vidloc "$@"
