#!/usr/bin/env bash
set -euo pipefail

python -m model_server.config-_generators.build_model_config

exec "$@"
