param(
    [string]$Results = "replication_results.jsonl",
    [string]$OutputDirectory = "results"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$resultsPath = (Resolve-Path -LiteralPath $Results).Path
$tasksPath = (Resolve-Path -LiteralPath (Join-Path $root "data\hylix_eval_tasks.jsonl")).Path
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$outputPath = (Resolve-Path -LiteralPath $OutputDirectory).Path

docker build -f (Join-Path $root "Dockerfile.evaluator") -t hylix-evaluator:1 $root
docker run --rm `
  --network none `
  --read-only `
  --pids-limit 64 `
  --memory 3g `
  --cpus 2 `
  --tmpfs /tmp:rw,noexec,nosuid,size=256m `
  --mount "type=bind,source=$tasksPath,target=/data/tasks.jsonl,readonly" `
  --mount "type=bind,source=$resultsPath,target=/data/results.jsonl,readonly" `
  --mount "type=bind,source=$outputPath,target=/output" `
  hylix-evaluator:1 `
  --tasks /data/tasks.jsonl `
  --results /data/results.jsonl `
  --output-dir /output `
  --execute-code
