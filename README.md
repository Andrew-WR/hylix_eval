# Hylix behavioral evaluation kit

This repo lets invited researchers compare normal Qwen-3.5-9B inference with
Hylix on new prompts or rerun a 20 code + math task held-out evaluation. Normal
and Hylix generations use the same prompt, decoding settings, and random seed.

This is a behavioral replication kit. The Hylix probe and training pipeline
remain private and are not required to evaluate the deployed method.

Overall, you will find Hylix to improve coding reasoning much more than math.
## What you need

- Python 3.10 or newer. The API client has no third-party dependencies.
- A RunPod endpoint ID and endpoint-restricted API key supplied privately.
- Docker only for safely scoring the HumanEval outputs.

Do not commit the API key.

## 1. Set credentials

macOS/Linux:

```bash
export RUNPOD_ENDPOINT_ID="provided-endpoint-id"
export RUNPOD_API_KEY="provided-restricted-key"
```

Windows PowerShell:

```powershell
$env:RUNPOD_ENDPOINT_ID="provided-endpoint-id"
$env:RUNPOD_API_KEY="provided-restricted-key"
```

Check access:

```bash
python hylix_client.py info
```

## 2. Try one prompt

```bash
python hylix_client.py compare --profile math --prompt "Solve x^2 - 5x + 6 = 0." --max-new-tokens 2048
```

Use `--profile raw` for general reasoning prompts, `math` for short-answer math,
and `code` for HumanEval-style function-completion prompts.

The client displays both generations and their token counts. `hylix_events` is
the number of intervention pulses during a Hylix generation.

## 3. Rerun the original held-out evaluation

The sample run contains 20 paired tasks and permits up to 8192 output tokens per
condition. It can take several hours and consumes a lot of endpoint credits, so please start with a smaller sample size. Start with
two tasks if desired:

```bash
python hylix_client.py benchmark --tasks data/hylix_eval_tasks.jsonl --mode compare --limit 2 --output smoke_results.jsonl
```

Full run:

```bash
python hylix_client.py benchmark --tasks data/hylix_eval_tasks.jsonl --mode compare --output replication_results.jsonl
```

The benchmark is checkpointed after every task. Rerun the identical command to
resume after an interruption.

Do not reuse an output file after changing mode, sampling parameters, task
manifest, or deployment. Start a new result file instead.

## 4. Score the results

HumanEval scoring executes model-generated Python. Use the supplied isolated
Docker evaluator, not your ordinary workstation Python environment. Read
`SECURITY.md` first.

Windows PowerShell:

```powershell
.\run_evaluation.ps1 -Results replication_results.jsonl -OutputDirectory results
```

macOS/Linux:

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh replication_results.jsonl results
```

Outputs:

- `results/summary.csv`: aggregate accuracy and token counts;
- `results/task_scores.jsonl`: task-level results and format diagnostics;
- `results/run_metadata.json`: model revision, Hylix fingerprint, and coverage.

Compare `results/summary.csv` with `reported_results.csv`. A fresh run is not
expected to reproduce every completion exactly; see `EXPERIMENT.md`.

## Run a new dataset

Create a JSONL file with one task per line:

```json
{"task_id":"math-1","domain":"math","split":"test","prompt":"Solve 2x + 3 = 11.","reference_answer":"4"}
```

Then collect paired outputs:

```bash
python hylix_client.py benchmark --tasks my_tasks.jsonl --mode compare --output my_results.jsonl
```

The included scorer supports the supplied MATH and HumanEval schemas. For other
domains, use the appropriate domain-specific evaluator on the generated JSONL.

## Files

- `hylix_client.py`: standard-library RunPod client and resumable collector.
- `data/hylix_eval_tasks.jsonl`: exact held-out task manifest.
- `score_results.py`: math and HumanEval scorer.
- `Dockerfile.evaluator`: isolated scoring environment.
- `reported_results.csv`: historical pilot result.
- `reported_sensitivity.csv`: historical PPL/KL/RULER guardrails.
- `EXPERIMENT.md`: scope, settings, and limitations.
- `SECURITY.md`: untrusted-code precautions.
- `kit_manifest.json`: task counts and hashes for the critical public files.
