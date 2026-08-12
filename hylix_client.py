#!/usr/bin/env python3
"""Public client for a private Hylix RunPod Serverless endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"})


def normalize_endpoint_id(value: str) -> str:
    """Accept either a RunPod endpoint ID or a copied queue API URL."""
    candidate = str(value).strip().rstrip("/")
    if candidate.startswith("ttps://"):
        candidate = "h" + candidate
    if "://" in candidate:
        parsed = urllib.parse.urlparse(candidate)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[0] != "v2":
            raise ValueError(
                "RunPod URL must look like https://api.runpod.ai/v2/ENDPOINT_ID"
            )
        candidate = parts[1]
    if not candidate or "/" in candidate:
        raise ValueError("RUNPOD_ENDPOINT_ID must contain a RunPod endpoint ID")
    return candidate


def paired_task_seed(base_seed: int, task_id: object) -> int:
    payload = "\x1f".join((str(task_id), "paired"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return int(base_seed) + int(digest[:8], 16) % 1_000_000


def request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RunPod returned HTTP {exc.code}: {detail}") from exc


class HylixClient:
    def __init__(self, endpoint_id: str, api_key: str, poll_seconds: float = 5.0):
        self.endpoint_id = normalize_endpoint_id(endpoint_id)
        self.base = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        self.api_key = api_key
        self.poll_seconds = poll_seconds

    def cancel(self, job_id: str) -> None:
        request_json("POST", f"{self.base}/cancel/{job_id}", self.api_key)

    def run(self, value: dict[str, Any], timeout_seconds: float = 1800) -> dict[str, Any]:
        submitted = request_json(
            "POST", f"{self.base}/run", self.api_key, {"input": value}
        )
        job_id = str(submitted.get("id", ""))
        if not job_id:
            raise RuntimeError(f"RunPod did not return a job id: {submitted}")
        print(f"Hylix job {job_id} submitted.", file=sys.stderr, flush=True)
        started = time.monotonic()
        last_status = None
        try:
            while True:
                status = request_json(
                    "GET", f"{self.base}/status/{job_id}", self.api_key
                )
                state = str(status.get("status", "UNKNOWN"))
                if state != last_status:
                    print(f"Hylix job status: {state}", file=sys.stderr, flush=True)
                    last_status = state
                if state in TERMINAL:
                    if state != "COMPLETED":
                        raise RuntimeError(f"Hylix job ended as {state}: {status}")
                    output = status.get("output")
                    if not isinstance(output, dict):
                        raise RuntimeError(f"Hylix returned an invalid output: {status}")
                    return output
                if time.monotonic() - started > timeout_seconds:
                    self.cancel(job_id)
                    raise TimeoutError(
                        f"Hylix job exceeded {timeout_seconds:g} seconds and was cancelled"
                    )
                time.sleep(self.poll_seconds)
        except KeyboardInterrupt:
            self.cancel(job_id)
            print(f"Cancelled Hylix job {job_id}.", file=sys.stderr)
            raise


def common_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "profile": args.profile,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(str(json.loads(line)["task_id"]))
    return result


def run_benchmark(
    client: HylixClient, args: argparse.Namespace
) -> None:
    tasks = []
    with Path(args.tasks).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    if args.split:
        tasks = [row for row in tasks if row.get("split") == args.split]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    destination = Path(args.output)
    completed = load_completed(destination)
    pending = [row for row in tasks if str(row["task_id"]) not in completed]
    for index, task in enumerate(pending, 1):
        domain = str(task["domain"])
        operation = "compare" if args.mode == "compare" else "generate"
        payload = {
            "operation": operation,
            "mode": args.mode if operation == "generate" else "hylix",
            "profile": domain,
            "prompt": str(task["prompt"]),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": paired_task_seed(args.seed, task["task_id"]),
        }
        print(
            f"Benchmark {index}/{len(pending)}: {task['task_id']}",
            file=sys.stderr,
            flush=True,
        )
        output = client.run(payload, args.timeout)
        append_jsonl(destination, {
            "task_id": task["task_id"],
            "domain": domain,
            "split": task.get("split"),
            "reference_answer": task.get("reference_answer"),
            "result": output,
        })
    print(f"Saved {len(tasks)} requested task results to {destination}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Call the Hylix demo endpoint")
    value.add_argument(
        "--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID")
    )
    value.add_argument(
        "--api-key", default=os.environ.get("RUNPOD_API_KEY"),
        help=argparse.SUPPRESS,
    )
    value.add_argument("--poll-seconds", type=float, default=5.0)
    value.add_argument("--timeout", type=float, default=1800.0)
    commands = value.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    for command in ("generate", "compare"):
        child = commands.add_parser(command)
        child.add_argument("--prompt", required=True)
        if command == "generate":
            child.add_argument("--mode", choices=("normal", "hylix"), default="hylix")
        child.add_argument("--profile", choices=("raw", "math", "code"), default="raw")
        child.add_argument("--max-new-tokens", type=int, default=2048)
        child.add_argument("--temperature", type=float, default=0.6)
        child.add_argument("--top-p", type=float, default=0.95)
        child.add_argument("--top-k", type=int, default=20)
        child.add_argument("--seed", type=int, default=20260801)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--tasks", required=True)
    benchmark.add_argument("--output", default="hylix_benchmark_results.jsonl")
    benchmark.add_argument("--mode", choices=("compare", "normal", "hylix"), default="compare")
    benchmark.add_argument("--split", choices=("train", "val", "test"), default="test")
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument("--max-new-tokens", type=int, default=8192)
    benchmark.add_argument("--temperature", type=float, default=0.6)
    benchmark.add_argument("--top-p", type=float, default=0.95)
    benchmark.add_argument("--top-k", type=int, default=20)
    benchmark.add_argument("--seed", type=int, default=20260801)
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.endpoint_id or not args.api_key:
        raise SystemExit(
            "Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY before running the client."
        )
    client = HylixClient(args.endpoint_id, args.api_key, args.poll_seconds)
    if args.command == "benchmark":
        run_benchmark(client, args)
        return
    if args.command == "info":
        payload = {"operation": "info"}
    else:
        payload = {"operation": args.command, **common_payload(args)}
        if args.command == "generate":
            payload["mode"] = args.mode
    result = client.run(payload, args.timeout)
    if args.command == "compare":
        print("\n===== NORMAL =====\n")
        print(result["normal"]["text"])
        print("\n===== HYLIX =====\n")
        print(result["hylix"]["text"])
        print("\n===== METADATA =====\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
