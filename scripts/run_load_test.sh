#!/usr/bin/env bash
set -euo pipefail

users="${1:-20}"
spawn_rate="${2:-10}"
run_time="${3:-60s}"
output_prefix="${4:-reports/locust/gateway}"
mode="${5:-scalar}"

case "$mode" in
  scalar)
    user_class="SentimentGatewayUser"
    ;;
  batch)
    user_class="SentimentBatchUser"
    ;;
  *)
    echo "mode must be scalar or batch" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$output_prefix")"
uv run locust \
  -f locustfile.py \
  "$user_class" \
  --headless \
  --users "$users" \
  --spawn-rate "$spawn_rate" \
  --run-time "$run_time" \
  --csv="$output_prefix" \
  --html="${output_prefix}.html" \
  --only-summary
