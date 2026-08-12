#!/usr/bin/env sh
set -eu

RESULTS="${1:-replication_results.jsonl}"
OUTPUT="${2:-results}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$OUTPUT"

docker build -f "$ROOT/Dockerfile.evaluator" -t hylix-evaluator:1 "$ROOT"
docker run --rm \
  --network none \
  --read-only \
  --pids-limit 64 \
  --memory 3g \
  --cpus 2 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --mount "type=bind,source=$(realpath "$ROOT/data/hylix_eval_tasks.jsonl"),target=/data/tasks.jsonl,readonly" \
  --mount "type=bind,source=$(realpath "$RESULTS"),target=/data/results.jsonl,readonly" \
  --mount "type=bind,source=$(realpath "$OUTPUT"),target=/output" \
  hylix-evaluator:1 \
  --tasks /data/tasks.jsonl \
  --results /data/results.jsonl \
  --output-dir /output \
  --execute-code
