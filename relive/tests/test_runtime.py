"""Closed runtime schema, acquisition, GT boundary and evaluation tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.acquisition import acquire, expand_candidate
from relive.audits import audit_runtime_imports
from relive.data.adapters import STGAdapter, task_support
from relive.data.frames import timing_summary
from relive.data.schemas import FIELD_SOURCES, RuntimeInputError, load_runtime


class RuntimeSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.new("RGB", (8, 8), "blue").save(self.root / "f.png")

    def tearDown(self):
        self.temp.cleanup()

    def runtime(self, row):
        path = self.root / "runtime.jsonl"
        payload = json.dumps(row, separators=(",", ":")) + "\n"
        path.write_text(payload, encoding="utf-8")
        sidecar = {"schema_version": "relive-runtime-v1", "source_kind": "synthetic",
                   "runtime_sha256": hashlib.sha256(payload.encode()).hexdigest(), "field_sources": FIELD_SOURCES}
        Path(str(path) + ".provenance.json").write_text(json.dumps(sidecar), encoding="utf-8")
        return path

    def row(self, task="claim_verification"):
        row = {"sample_id": "s", "task": task, "question": "What is visible?",
               "frames": [{"frame_id": "f0", "path": "f.png", "order": 0},
                          {"frame_id": "f1", "path": "f.png", "order": 1}],
               "metadata": {"question_scope": "single_action"}}
        if task == "claim_verification":
            row["target_claim"] = {"claim_id": "c", "text": "A visible action occurs."}
        return row

    def test_loads_public_runtime_fields_without_invented_timing(self):
        sample = load_runtime(self.runtime(self.row()))[0]
        self.assertEqual(sample.frames[0].order, 0)
        self.assertIsNone(sample.frames[0].timestamp)
        self.assertFalse(timing_summary(sample.frames)["seconds_available"])
        self.assertIn("No second-level", timing_summary(sample.frames)["limitation"])

    def test_action_qa_requires_frozen_requirements_and_claim_flag_is_not_accepted(self):
        with self.assertRaisesRegex(RuntimeInputError, "action_qa requires nonempty required_claims"):
            load_runtime(self.runtime(self.row("action_qa")))
        row = self.row("action_qa")
        row["required_claims"] = [{"claim_id": "required", "text": "A visible action occurs."}]
        self.assertEqual(load_runtime(self.runtime(row))[0].required_claims[0].claim_id, "required")
        row["required_claims"][0]["required_for_question"] = False
        with self.assertRaisesRegex(RuntimeInputError, "non-runtime fields"):
            load_runtime(self.runtime(row))

    def test_gt_and_unknown_fields_cannot_enter_runtime(self):
        for field in ("answer", "bbox", "mask", "ground_truth", "struc_info"):
            row = self.row()
            row[field] = "forbidden"
            with self.subTest(field=field), self.assertRaises(RuntimeInputError):
                load_runtime(self.runtime(row))
        row = self.row()
        row["metadata"]["hidden_answer"] = "forbidden"
        with self.assertRaises(RuntimeInputError):
            load_runtime(self.runtime(row))

    def test_hash_attestation_and_timestamp_sources_are_enforced(self):
        path = self.runtime(self.row())
        path.write_text(path.read_text() + " ", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeInputError, "SHA-256"):
            load_runtime(path)
        row = self.row()
        row["frames"][0]["timestamp"] = 1.0
        row["frames"][0]["timestamp_source"] = "guessed_fps"
        row["frames"][0]["source_reference"] = "made up"
        with self.assertRaises(RuntimeInputError):
            load_runtime(self.runtime(row))

    def test_unsupported_task_is_preserved_for_explicit_runner_result(self):
        sample = load_runtime(self.runtime(self.row("stg")))[0]
        self.assertEqual(sample.task, "stg")
        self.assertEqual(task_support("stg")["status"], "UNSUPPORTED")
        self.assertFalse(STGAdapter().status()["official_runtime_connected"])

    def test_acquisition_is_deterministic_and_gt_independent(self):
        row = self.row()
        row["frames"] += [{"frame_id": "f2", "path": "f.png", "order": 2},
                          {"frame_id": "f3", "path": "f.png", "order": 3}]
        sample = load_runtime(self.runtime(row))[0]
        windows = acquire(sample, {"method": "sliding_windows", "window_size": 2, "stride": 1, "max_candidates": 3})
        self.assertEqual([c.acquisition_rank for c in windows], [0, 1, 2])
        self.assertEqual([c.rank_source for c in windows], ["chronological"] * 3)
        self.assertTrue(all(not c.provenance["semantic_relevance_scored"] for c in windows))
        uniform = acquire(sample, {"method": "uniform", "window_size": 2, "stride": 1, "max_candidates": 1})
        self.assertEqual(uniform[0].rank_source, "uniform")
        expanded = expand_candidate(sample, windows[1], 1)
        self.assertEqual(expanded.frame_ids, ("f0", "f1", "f2", "f3"))

    def test_runtime_import_boundary_static_audit_passes(self):
        self.assertEqual(audit_runtime_imports()["status"], "PASS")
