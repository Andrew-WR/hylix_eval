#!/usr/bin/env python3
"""Score paired Hylix endpoint results against the held-out task manifest."""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import io
import json
import multiprocessing as mp
import os
import queue
import re
import signal
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


FINAL_PATTERN = re.compile(
    r"(?im)^\s*\**final(?:\s+answer)?\**\s*:\**\s*(.+?)\s*$"
)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|<>]+\|>")
WORD_PATTERN = re.compile(r"[a-z0-9]+")
BOX_START_PATTERN = re.compile(r"\\(?:boxed|fbox)\{")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def normalize_answer(value: str) -> str:
    value = SPECIAL_TOKEN_PATTERN.sub("", value)
    return " ".join(WORD_PATTERN.findall(value.casefold()))


def terminal_boxed_answer(text: str) -> str | None:
    lowered = text.casefold()
    thinking_open = lowered.rfind("<think>")
    thinking_close = lowered.rfind("</think>")
    if thinking_open >= 0 and thinking_close < thinking_open:
        return None
    start = thinking_close + len("</think>") if thinking_close >= 0 else 0
    candidates = []
    for match in BOX_START_PATTERN.finditer(text, start):
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            depth += int(text[index] == "{")
            depth -= int(text[index] == "}")
            index += 1
        if depth:
            continue
        suffix = SPECIAL_TOKEN_PATTERN.sub("", text[index:])
        suffix = re.sub(r"</?s>", "", suffix, flags=re.IGNORECASE)
        suffix = suffix.replace(r"\)", "").replace(r"\]", "")
        if not suffix.strip(" \t\r\n$.,;:!?*_`"):
            candidates.append(text[match.end() : index - 1].strip())
    return candidates[-1] if candidates else None


def extract_final_answer(text: str) -> str | None:
    matches = list(FINAL_PATTERN.finditer(text))
    lowered = text.casefold()
    thinking_open = lowered.rfind("<think>")
    thinking_close = lowered.rfind("</think>")
    if thinking_open >= 0:
        matches = (
            [] if thinking_close < thinking_open
            else [match for match in matches if match.start() >= thinking_close]
        )
    return matches[-1].group(1).strip() if matches else terminal_boxed_answer(text)


def math_correct(text: str, expected: str) -> tuple[bool, str | None]:
    answer = extract_final_answer(text)
    if answer is None:
        return False, None
    observed = normalize_answer(answer)
    target = normalize_answer(expected)
    for prefix in ("the answer is ", "answer is ", "boxed "):
        if observed.startswith(prefix):
            observed = observed[len(prefix) :]
            break
    return bool(observed and target and observed == target), answer


def extract_code_completion(text: str) -> str:
    marker = text.casefold().rfind("final_code:")
    value = text[marker + len("final_code:") :] if marker >= 0 else text
    fenced = re.search(
        r"```(?:python|py)?\s*\n(?P<code>.*?)```",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        value = fenced.group("code")
    value = value.replace("<|im_end|>", "").strip("\r\n")
    return value + ("\n" if value else "")


def completion_format(completion: str, entry_point: str) -> tuple[bool, str]:
    value = completion.lstrip("\r\n")
    if not value.strip():
        return False, "empty_completion"
    if any(marker in value for marker in ("```", "<|", "FINAL_CODE:")):
        return False, "generation_marker_or_fence"
    if not value[0].isspace():
        return False, "not_an_indented_prompt_continuation"
    repeated = re.compile(
        rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    )
    if repeated.search(value):
        return False, "repeats_supplied_function"
    return True, "valid_completion_only_format"


class RenameEntry(ast.NodeTransformer):
    def __init__(self, original: str, replacement: str):
        self.original = original
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.original:
            return ast.copy_location(
                ast.Name(id=self.replacement, ctx=node.ctx), node
            )
        return node


def entry_call_arguments(
    function: ast.FunctionDef,
) -> tuple[list[ast.expr], list[ast.keyword]]:
    arguments: list[ast.expr] = [
        ast.Name(id=value.arg, ctx=ast.Load())
        for value in (*function.args.posonlyargs, *function.args.args)
    ]
    if function.args.vararg is not None:
        arguments.append(ast.Starred(
            value=ast.Name(id=function.args.vararg.arg, ctx=ast.Load()),
            ctx=ast.Load(),
        ))
    keywords = [
        ast.keyword(arg=value.arg, value=ast.Name(id=value.arg, ctx=ast.Load()))
        for value in function.args.kwonlyargs
    ]
    if function.args.kwarg is not None:
        keywords.append(ast.keyword(
            arg=None,
            value=ast.Name(id=function.args.kwarg.arg, ctx=ast.Load()),
        ))
    return arguments, keywords


def normalize_standalone_completion(
    prompt: str, completion: str, entry_point: str,
) -> tuple[str | None, str]:
    valid, _ = completion_format(completion, entry_point)
    if valid:
        return completion, "already_valid_completion"
    try:
        prompt_tree = ast.parse(prompt)
        generated_tree = ast.parse(completion)
    except SyntaxError as exc:
        return None, f"unparseable_python:{exc.msg}"
    supplied = next((
        node for node in reversed(prompt_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ), None)
    generated = next((
        node for node in reversed(generated_tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ), None)
    if supplied is None or generated is None:
        return None, "entry_function_not_found"
    occupied = {
        node.name for node in generated_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    helper = f"_normalized_{entry_point}"
    suffix = 2
    while helper in occupied:
        helper = f"_normalized_{entry_point}_{suffix}"
        suffix += 1
    safe_support = (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign,
    )
    support = [
        copy.deepcopy(node) for node in generated_tree.body
        if isinstance(node, safe_support)
        and not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == entry_point
        )
    ]
    nested = copy.deepcopy(generated)
    nested.name = helper
    renamer = RenameEntry(entry_point, helper)
    support = [renamer.visit(node) for node in support]
    nested = renamer.visit(nested)
    arguments, keywords = entry_call_arguments(supplied)
    invoke = ast.Return(value=ast.Call(
        func=ast.Name(id=helper, ctx=ast.Load()),
        args=arguments,
        keywords=keywords,
    ))
    module = ast.Module(body=[*support, nested, invoke], type_ignores=[])
    ast.fix_missing_locations(module)
    normalized = textwrap.indent(ast.unparse(module), "    ") + "\n"
    valid, reason = completion_format(normalized, entry_point)
    if not valid:
        return None, f"normalizer_produced_invalid_format:{reason}"
    return normalized, "standalone_entry_wrapped"


def build_program(task: dict[str, Any], completion: str) -> str:
    metadata = task["metadata"]
    return (
        str(task["prompt"]) + completion + "\n"
        + str(metadata["test"]) + "\n"
        + f"check({metadata['entry_point']})"
    )


def execute_program(program: str, result: Any) -> None:
    try:
        if hasattr(os, "setsid"):
            os.setsid()
        os.environ.clear()
        os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            import resource

            two_gib = 2 * 1024**3
            resource.setrlimit(resource.RLIMIT_AS, (two_gib, two_gib))
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2)
            )
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except (ImportError, OSError, ValueError):
            pass
        with tempfile.TemporaryDirectory() as temporary:
            previous = os.getcwd()
            os.chdir(temporary)
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    exec(program, {})
            finally:
                os.chdir(previous)
        result.put({"passed": True, "result": "passed"})
    except BaseException as exc:
        result.put({
            "passed": False,
            "result": f"failed: {type(exc).__name__}: {exc}",
        })


def check_program(program: str, timeout: float) -> dict[str, Any]:
    context = mp.get_context("spawn")
    result = context.Queue(maxsize=1)
    process = context.Process(target=execute_program, args=(program, result))
    process.start()
    process.join(timeout + 1.0)
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()
        process.join()
    try:
        return dict(result.get(timeout=0.2))
    except queue.Empty:
        return {"passed": False, "result": "timed out"}
    finally:
        result.close()


def score_code(
    task: dict[str, Any], text: str, timeout: float, execute: bool,
) -> dict[str, Any]:
    completion = extract_code_completion(text)
    entry = str(task["metadata"]["entry_point"])
    format_valid, format_reason = completion_format(completion, entry)
    if not execute:
        return {
            "completion": completion,
            "strict_correct": None,
            "functional_correct": None,
            "normalized_correct": None,
            "format_valid": format_valid,
            "format_reason": format_reason,
            "execution": "not executed",
        }
    direct = check_program(build_program(task, completion), timeout)
    normalized, normalization_reason = normalize_standalone_completion(
        str(task["prompt"]), completion, entry
    )
    if normalized is None:
        normalized_outcome = {"passed": False, "result": "normalization unavailable"}
        normalized_format = False
    elif normalization_reason == "already_valid_completion":
        normalized_outcome = direct
        normalized_format = format_valid
    else:
        normalized_outcome = check_program(build_program(task, normalized), timeout)
        normalized_format, _ = completion_format(normalized, entry)
    functional = bool(direct["passed"])
    normalized_functional = bool(normalized_outcome["passed"])
    return {
        "completion": completion,
        "strict_correct": bool(functional and format_valid),
        "functional_correct": functional,
        "normalized_correct": bool(normalized_functional and normalized_format),
        "format_valid": format_valid,
        "format_reason": format_reason,
        "execution": direct["result"],
        "normalization_reason": normalization_reason,
        "normalized_execution": normalized_outcome["result"],
    }


def extract_conditions(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{row.get('task_id')}: missing result object")
    paired = {
        condition: result[condition]
        for condition in ("normal", "hylix")
        if isinstance(result.get(condition), dict)
    }
    if paired:
        return paired
    generation = result.get("generation")
    if isinstance(generation, dict) and generation.get("mode") in {"normal", "hylix"}:
        return {str(generation["mode"]): generation}
    raise ValueError(f"{row.get('task_id')}: no normal or hylix generation found")


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Hylix replication results")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute-code", action="store_true")
    parser.add_argument("--code-timeout", type=float, default=3.0)
    args = parser.parse_args()
    tasks = {str(row["task_id"]): row for row in read_jsonl(args.tasks)}
    raw_results = read_jsonl(args.results)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details = []
    metadata = defaultdict(set)
    seen = set()
    for row in raw_results:
        task_id = str(row["task_id"])
        if task_id not in tasks:
            raise ValueError(f"unknown task_id in results: {task_id}")
        task = tasks[task_id]
        if str(row.get("domain")) != str(task["domain"]):
            raise ValueError(f"domain mismatch for {task_id}")
        for condition, generation in extract_conditions(row).items():
            key = (task_id, condition)
            if key in seen:
                raise ValueError(f"duplicate result for {task_id}/{condition}")
            seen.add(key)
            endpoint = row["result"]
            for field in ("model", "model_revision", "probe_fingerprint", "version"):
                if endpoint.get(field) is not None:
                    metadata[field].add(str(endpoint[field]))
            text = str(generation.get("text", ""))
            domain = str(task["domain"])
            if domain == "math":
                correct, answer = math_correct(text, str(task["reference_answer"]))
                score = {
                    "strict_correct": correct,
                    "functional_correct": correct,
                    "normalized_correct": correct,
                    "observed_answer": answer,
                }
            elif domain == "code":
                score = score_code(
                    task, text, args.code_timeout, args.execute_code
                )
            else:
                raise ValueError(f"unsupported scored domain: {domain}")
            details.append({
                "task_id": task_id,
                "condition": condition,
                "domain": domain,
                "output_tokens": int(generation.get("output_tokens", 0)),
                "hylix_events": generation.get("hylix_events"),
                **score,
            })
    detail_path = output / "task_scores.jsonl"
    with detail_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    conditions = sorted({str(row["condition"]) for row in details})
    for condition in conditions:
        for domain in ("math", "code", "overall"):
            selected = [
                row for row in details
                if row["condition"] == condition
                and (domain == "overall" or row["domain"] == domain)
            ]
            scored = [row for row in selected if row["strict_correct"] is not None]
            summary.append({
                "condition": condition,
                "domain": domain,
                "tasks": len(selected),
                "scored_tasks": len(scored),
                "strict_accuracy": mean([
                    float(row["strict_correct"]) for row in scored
                ]),
                "functional_accuracy": mean([
                    float(row["functional_correct"]) for row in scored
                ]),
                "normalized_accuracy": mean([
                    float(row["normalized_correct"]) for row in scored
                ]),
                "mean_output_tokens": mean([
                    float(row["output_tokens"]) for row in selected
                ]),
            })
    fields = list(summary[0]) if summary else []
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    run_metadata = {key: sorted(values) for key, values in metadata.items()}
    run_metadata.update({
        "task_manifest_count": len(tasks),
        "result_task_count": len({row["task_id"] for row in details}),
        "code_executed": bool(args.execute_code),
    })
    write_json(output / "run_metadata.json", run_metadata)
    print("\ncondition  domain   scored/tasks  strict  functional  normalized  mean_tokens")
    for row in summary:
        metric = lambda key: (
            "NA" if row[key] is None else f"{100 * float(row[key]):.1f}%"
        )
        tokens = (
            "NA" if row["mean_output_tokens"] is None
            else f"{float(row['mean_output_tokens']):.1f}"
        )
        print(
            f"{row['condition']:<10} {row['domain']:<8} "
            f"{row['scored_tasks']:>3}/{row['tasks']:<3} "
            f"{metric('strict_accuracy'):>8} "
            f"{metric('functional_accuracy'):>11} "
            f"{metric('normalized_accuracy'):>11} {tokens:>12}"
        )
    print(f"\nWrote {detail_path}, summary.csv, and run_metadata.json")


if __name__ == "__main__":
    main()
