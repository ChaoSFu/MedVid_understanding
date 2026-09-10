"""Regression test for the read-only Phase 1.5 diagnostic utility."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.config import load_config
from relive.runner import run


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "phase15_certificate_diagnose.py"
CONFIG = ROOT / "configs" / "mock_smoke.yaml"
RUNTIME = ROOT / "examples" / "synthetic_runtime.jsonl"


class Phase15DiagnosticTests(unittest.TestCase):
    def test_reads_completed_mock_artifacts_without_model_execution_or_run_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir, cache_dir, output_dir = root / "run", root / "cache", root / "phase15"
            run(load_config(CONFIG), RUNTIME, run_dir, cache_dir=cache_dir, max_samples=1)
            before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*.json")}
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), "--run-dir", str(run_dir), "--runtime", str(RUNTIME),
                 "--cache-dir", str(cache_dir), "--output-dir", str(output_dir)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output_dir / "diagnostic_report.json").read_text())
            self.assertTrue(report["read_only"])
            self.assertEqual(report["model_calls_made"], 0)
            self.assertEqual(report["summary"]["candidate_count"], 1)
            self.assertEqual(report["policy_decomposition"]["semantic_only"]["certificate_distribution"], {"VERIFIED": 1})
            self.assertTrue(list((output_dir / "contact_sheets").glob("*.png")))
            self.assertTrue((output_dir / "diagnostic_report.md").is_file())
            self.assertTrue((output_dir / "summary_table.csv").is_file())
            after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*.json")}
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
