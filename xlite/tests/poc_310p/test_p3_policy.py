"""CPU policy validation, without importing torch or the NPU extension."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "xlite" / "p3.py"
SPEC = importlib.util.spec_from_file_location("p3_policy", SOURCE)
P3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P3)


class PolicyTests(unittest.TestCase):
    def policy(self):
        return {"schema": 1, "fingerprint": {"id": "test"}, "entries": [
            {"m": 1, "n": 151936, "k": 2048, "chunk": 24576, "direct": True, "enabled": True}]}

    def test_valid(self):
        self.assertTrue(P3.validate_policy(self.policy(), {"id": "test"}))

    def test_mismatch(self):
        self.assertFalse(P3.validate_policy(self.policy(), {"id": "other-build"}))

    def test_invalid(self):
        for key, value in (("m", 21), ("chunk", 1000000), ("k", 123), ("direct", "false")):
            policy = self.policy()
            policy["entries"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                P3.validate_policy(policy, {"id": "test"})

    def test_duplicates(self):
        policy = self.policy()
        policy["entries"] *= 2
        with self.assertRaises(ValueError):
            P3.validate_policy(policy, {"id": "test"})

    def test_chunk_coverage_and_storage_offsets(self):
        for m in (1, 8, 20):
            for chunk in P3.CHUNKS:
                n = 151936
                ranges = [(offset, min(chunk, n - offset)) for offset in range(0, n, chunk)]
                self.assertEqual(sum(width for _, width in ranges), n)
                for offset, width in ranges:
                    self.assertLessEqual(offset + (m - 1) * n + width, m * n)
                for (offset, width), (next_offset, _) in zip(ranges, ranges[1:]):
                    self.assertEqual(offset + width, next_offset)

    def test_screen_falls_back_and_reruns_only_failures(self):
        source = Path(__file__).with_name("tune_p3_matmul.py")
        spec = importlib.util.spec_from_file_location("p3_tuner", source)
        tuner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tuner)
        calls = []

        def fake_run(command, **kwargs):
            shape = json.loads(command[command.index("--worker") + 1])
            target = Path(command[command.index("--output") + 1])
            calls.append(shape)
            rejected = shape["chunk"] > 12288
            result = {"spec": shape, "fingerprint": {"id": "test"},
                      "status": "failed" if rejected else "passed", "error": "workspace budget"}
            result["median_ms"] = 2 if shape["mode"] == "legacy" else (1 if shape["m"] == 1 else 3)
            target.write_text(json.dumps(result))
            return SimpleNamespace(returncode=1 if rejected else 0)

        with tempfile.TemporaryDirectory() as directory, patch.object(tuner.subprocess, "run", fake_run):
            args = [str(source), "--report-dir", directory]
            with patch.object(tuner.sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tuner.main(), 0)
            policy = json.loads((Path(directory) / "policy.json").read_text())
            self.assertTrue(any(e["enabled"] for e in policy["entries"]))
            self.assertTrue(all(not e["enabled"] for e in policy["entries"] if e["m"] != 1))
            calls.clear()
            with patch.object(tuner.sys, "argv", args + ["--rerun-failed"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(tuner.main(), 0)
            self.assertTrue(calls)
            self.assertTrue(all(call["chunk"] > 12288 for call in calls))


if __name__ == "__main__":
    unittest.main()
