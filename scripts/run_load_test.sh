#!/usr/bin/env bash
set -euo pipefail

users="${1:-20}"
spawn_rate="${2:-10}"
run_time="${3:-60s}"
output_prefix="${4:-reports/locust/gateway}"

mkdir -p "$(dirname "$output_prefix")"
uv run locust \
  -f locustfile.py \
  SentimentGatewayUser \
  --headless \
  --users "$users" \
  --spawn-rate "$spawn_rate" \
  --run-time "$run_time" \
  --csv="$output_prefix" \
  --html="${output_prefix}.html" \
  --only-summary
