#!/usr/bin/env bash
set -euo pipefail

python -m model_server.config_generators.build_model_config

exec "$@"
