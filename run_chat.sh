#!/bin/bash
# quick chat script

cd "$(dirname "$0")"
python -m nanochat.chat "$@"

