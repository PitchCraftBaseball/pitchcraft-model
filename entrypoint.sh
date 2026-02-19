#!/usr/bin/env bash
set -euo pipefail

python ./model_server/config-generators/build_model_config.py

exec "$@"
