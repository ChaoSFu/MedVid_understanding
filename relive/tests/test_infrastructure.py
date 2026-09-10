"""Cache and configuration behavior, independent of model hardware."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.backends.base import Backend, BackendError
from relive.config import load_config, validate_config
from relive.storage import ArtifactError, ArtifactStore, Budget, BudgetExceeded, CachedInference


class StubBackend(Backend):
    synthetic = True

    def __init__(self, failures=0, retries=0):
        super().__init__({"kind": "stub", "model": "stub", "revision": "v1", "max_retries": retries})
        self.failures = failures

    def infer(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise BackendError("MOCK_FIXTURE_ERROR", retryable=True)
        return '{"status":"SUPPORTED"}'


def request(path: Path, **changes):
    value = {"stage": "semantic", "prompt": "fixed prompt", "prompt_version": "p1",
             "image_paths": [str(path)], "frame_ids": ["f0"],
             "context": {"claim": "c", "region": [0.1, 0.1, 0.4, 0.4],
                         "intervention": "ORIGINAL", "preprocessing": {"resize": False}}}
    value.update(changes)
    return value


class ArtifactAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = self.root / "f.png"
        Image.new("RGB", (8, 6), (30, 80, 140)).save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def test_artifacts_are_write_once_and_complete_json_only(self):
        store = ArtifactStore(self.root / "artifacts")
        reference = store.put_json("stage", "item", {"x": 1})
        self.assertEqual(store.get_json(reference), {"x": 1})
        self.assertEqual(store.put_json("stage", "item", {"x": 1}), reference)
        with self.assertRaises(ArtifactError):
            store.put_json("stage", "item", {"x": 2})
        with self.assertRaises(ArtifactError):
            store.get_json(str(self.root / "outside.json"))

    def test_cache_identity_includes_bytes_and_protocol_inputs_but_not_policy(self):
        store = ArtifactStore(self.root / "cache")
        cached = CachedInference(StubBackend(), store)
        base = request(self.image)
        same_policy_change = deepcopy(base)
        same_policy_change["policy_name"] = "semantic_only"
        changed_region = deepcopy(base)
        changed_region["context"]["region"] = [0.2, 0.1, 0.4, 0.4]
        changed_prompt = deepcopy(base)
        changed_prompt["prompt"] = "other prompt"
        base_key = cached.cache_key(base)
        self.assertEqual(base_key, cached.cache_key(same_policy_change))
        self.assertNotEqual(base_key, cached.cache_key(changed_region))
        self.assertNotEqual(base_key, cached.cache_key(changed_prompt))
        self.image.write_bytes(self.image.read_bytes() + b"\n")
        self.assertNotEqual(base_key, cached.cache_key(base))

    def test_cache_replay_has_zero_new_calls_but_same_logical_budget_cost(self):
        store = ArtifactStore(self.root / "cache")
        backend = StubBackend()
        cached = CachedInference(backend, store)
        first_budget = Budget(4)
        first = cached.call(request(self.image), first_budget)
        self.assertFalse(first["cache_hit"])
        self.assertEqual((backend.calls, first_budget.calls, first_budget.new_calls), (1, 1, 1))
        replay_budget = Budget(4)
        replay = cached.call(request(self.image), replay_budget)
        self.assertTrue(replay["cache_hit"])
        self.assertEqual(backend.calls, 1)
        self.assertEqual((replay_budget.calls, replay_budget.new_calls, replay_budget.cache_hits), (1, 0, 1))

    def test_retry_records_every_technical_attempt_and_never_retries_semantics(self):
        store = ArtifactStore(self.root / "cache")
        backend = StubBackend(failures=1, retries=1)
        outcome = CachedInference(backend, store).call(request(self.image), Budget(4))
        self.assertEqual(outcome["execution_status"], "OK")
        self.assertEqual(len(outcome["attempts"]), 2)
        self.assertEqual([a["execution_status"] for a in outcome["attempts"]], ["INFERENCE_ERROR", "OK"])
        self.assertEqual(backend.calls, 2)

    def test_budget_refuses_new_request_before_model_execution(self):
        store = ArtifactStore(self.root / "cache")
        backend = StubBackend()
        cached = CachedInference(backend, store)
        budget = Budget(1)
        cached.call(request(self.image), budget)
        next_request = request(self.image, prompt="another request")
        with self.assertRaises(BudgetExceeded):
            cached.call(next_request, budget)
        self.assertEqual(backend.calls, 1)

    def test_incomplete_or_untrusted_raw_record_is_not_a_cache_success(self):
        store = ArtifactStore(self.root / "cache")
        backend = StubBackend()
        cached = CachedInference(backend, store)
        key = cached.cache_key(request(self.image))
        store.put_json(f"raw/{key}", "not-a-content-hash", {"incomplete": True})
        outcome = cached.call(request(self.image), Budget(4))
        self.assertEqual(outcome["execution_status"], "OK")
        self.assertFalse(outcome["cache_hit"])
        self.assertEqual(backend.calls, 1)


class ConfigTests(unittest.TestCase):
    def test_real_backend_requires_explicit_transport_settings_and_no_literal_secret(self):
        with self.assertRaises(ValueError):
            validate_config({"backend": {"kind": "openai_compatible"}})
        real = {
            "kind": "openai_compatible", "base_url": "https://model.example/v1",
            "endpoint_path": "/chat/completions", "model": "frozen-model", "revision": "r1",
            "credential_env": "RELIVE_KEY", "generation": {"temperature": 0.0},
            "image_order": "chronological", "frame_encoding": "png", "timeout_seconds": 1,
            "max_retries": 0, "extra": {},
        }
        self.assertEqual(validate_config({"backend": real, "policy": {"name": "semantic_spatial"}})["backend"]["model"],
                         "frozen-model")
        secret = deepcopy(real)
        secret["api_key"] = "not-permitted"
        with self.assertRaises(ValueError):
            validate_config({"backend": secret})

    def test_real_full_contrast_policy_fails_fast_without_declared_exclusivity(self):
        real = {
            "kind": "openai_compatible", "base_url": "https://model.example/v1",
            "endpoint_path": "/chat/completions", "model": "frozen-model", "revision": "r1",
            "credential_env": "RELIVE_KEY", "generation": {"temperature": 0.0},
            "image_order": "chronological", "frame_encoding": "png", "timeout_seconds": 1,
            "max_retries": 0, "extra": {},
        }
        with self.assertRaisesRegex(ValueError, "exclusivity source"):
            validate_config({"backend": real, "policy": {"name": "semantic_contrast_spatial"}})
        self.assertEqual(validate_config({"backend": real, "policy": {"name": "semantic_spatial"}})["policy"]["name"],
                         "semantic_spatial")

    def test_reviewed_qwen_smokes_disable_full_contrast_and_fixed_alternates(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("qwen35_9b_medvidu_claim_smoke.yaml", "qwen35_9b_medvidu_claim_smoke_no_thinking.yaml"):
            config = load_config(root / "configs" / name)
            self.assertEqual(config["backend"]["kind"], "local_hf")
            self.assertEqual(config["policy"]["name"], "semantic_spatial")
            self.assertEqual(config["spatial"]["alternate_regions"], [])

    def test_unknown_adaptation_action_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_ACTION"):
            validate_config({"adaptation": {"actions": ["FREEZE_MODEL"]}})
