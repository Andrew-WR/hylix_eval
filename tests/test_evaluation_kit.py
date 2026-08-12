import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SCORER = load("hylix_public_scorer", ROOT / "score_results.py")
CLIENT = load("hylix_public_client", ROOT / "hylix_client.py")


class EvaluationKitTests(unittest.TestCase):
    def test_endpoint_id_accepts_id_or_url(self):
        self.assertEqual(CLIENT.normalize_endpoint_id("abc123"), "abc123")
        self.assertEqual(
            CLIENT.normalize_endpoint_id(
                "https://api.runpod.ai/v2/abc123/runsync"
            ),
            "abc123",
        )

    def test_final_answer_is_post_thinking(self):
        text = "<think>FINAL: wrong</think>\nFINAL: 4<|im_end|>"
        passed, answer = SCORER.math_correct(text, "4")
        self.assertTrue(passed)
        self.assertEqual(answer, "4<|im_end|>")

    def test_extract_code_block(self):
        text = "<think>reasoning</think>\nFINAL_CODE:\n```python\n    return 4\n```"
        self.assertEqual(SCORER.extract_code_completion(text), "    return 4\n")

    def test_standalone_function_can_be_normalized(self):
        prompt = "def square(x):\n    \"\"\"Return x squared.\"\"\"\n"
        completion = "def square(x):\n    return x * x\n"
        normalized, reason = SCORER.normalize_standalone_completion(
            prompt, completion, "square"
        )
        self.assertEqual(reason, "standalone_entry_wrapped")
        self.assertIsNotNone(normalized)
        valid, _ = SCORER.completion_format(normalized, "square")
        self.assertTrue(valid)

    def test_end_to_end_math_summary(self):
        task = {
            "task_id": "m1", "domain": "math", "split": "test",
            "prompt": "2+2?", "reference_answer": "4", "metadata": {},
        }
        result = {
            "task_id": "m1", "domain": "math", "split": "test",
            "result": {
                "model": "test/model", "probe_fingerprint": "abc",
                "normal": {"text": "FINAL: 3", "output_tokens": 10},
                "hylix": {"text": "FINAL: 4", "output_tokens": 8},
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "tasks.jsonl"
            results = root / "results.jsonl"
            output = root / "scores"
            tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
            results.write_text(json.dumps(result) + "\n", encoding="utf-8")
            previous = sys.argv
            try:
                sys.argv = [
                    "score_results.py", "--tasks", str(tasks), "--results",
                    str(results), "--output-dir", str(output),
                ]
                SCORER.main()
            finally:
                sys.argv = previous
            summary = (output / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("hylix,overall,1,1,1.0,1.0,1.0,8.0", summary)
            self.assertIn("normal,overall,1,1,0.0,0.0,0.0,10.0", summary)


if __name__ == "__main__":
    unittest.main()
