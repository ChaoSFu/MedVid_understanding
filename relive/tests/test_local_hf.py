"""Local-HF checkpoint inspection and strict adapter tests without model weights."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.backends.inspection import checkpoint_metadata, inspect_local_hf
from relive.backends.local_hf import LocalHFBackend
from relive.config import validate_backend


class _FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    template = "{{ checkpoint_declared_template }}"
    last = None

    def __init__(self):
        self.chat_template = self.template
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if getattr(self, "raise_template_error", False):
            raise TypeError("unsupported contract")
        if kwargs["tokenize"]:
            return _FakeInputs(input_ids=[[1, 2]])
        return "rendered prompt"

    def __call__(self, **kwargs):
        self.process_call = kwargs
        return _FakeInputs(input_ids=[[1, 2]])

    def batch_decode(self, trimmed, **kwargs):
        self.trimmed = trimmed
        return ['{"status":"SUPPORTED"}']


class _ProbeTensor:
    """Minimal CPU-tensor-shaped fixture for processor-contract audits."""

    def __init__(self, value):
        self.value = value
        self.shape = self._shape(value)
        self.dtype = "fake"
        self.device = "cpu"

    @staticmethod
    def _shape(value):
        if not isinstance(value, list):
            return ()
        return (len(value),) + (_ProbeTensor._shape(value[0]) if value else ())

    def numel(self):
        def count(value):
            return sum(count(item) for item in value) if isinstance(value, list) else 1
        return count(self.value)

    def tolist(self):
        return self.value


class VisualProbeProcessor(FakeProcessor):
    """Fake visual processor that preserves two image input orders."""

    @staticmethod
    def _inputs(images):
        rows = [[1, image.height // 28, image.width // 28] for image in images]
        pixels = [[image.width, image.height, *image.getpixel((0, 0))] for image in images]
        return _FakeInputs(
            input_ids=_ProbeTensor([[1, 2, 3]]),
            pixel_values=_ProbeTensor(pixels),
            image_grid_thw=_ProbeTensor(rows),
        )

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["tokenize"]:
            images = [item["image"] for item in messages[0]["content"] if item["type"] == "image"]
            return self._inputs(images)
        return "rendered visual probe"

    def __call__(self, **kwargs):
        self.process_call = kwargs
        return self._inputs(kwargs["images"])


class MissingGridProbeProcessor(VisualProbeProcessor):
    @staticmethod
    def _inputs(images):
        result = VisualProbeProcessor._inputs(images)
        result.pop("image_grid_thw")
        return result


class FakeVisionModel:
    last = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        instance.path = path
        instance.load_kwargs = kwargs
        cls.last = instance
        return instance

    def to(self, device):
        self.placed = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


def fake_modules(processor: FakeProcessor):
    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            processor.load_path = path
            processor.load_kwargs = kwargs
            return processor

    torch = types.SimpleNamespace(
        bfloat16="bf16", float16="fp16", float32="fp32", __version__="fake-torch",
        inference_mode=lambda: contextlib.nullcontext(),
    )
    transformers = types.SimpleNamespace(
        AutoProcessor=FakeAutoProcessor, FakeVisionModel=FakeVisionModel, __version__="fake-transformers",
    )
    return {"torch": torch, "transformers": transformers}


class LocalHFTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"model_type": "fake_vl", "architectures": ["FakeVisionModel"]}))
        (self.model / "tokenizer_config.json").write_text(json.dumps({"chat_template": FakeProcessor.template}))
        (self.model / "processor_config.json").write_text(json.dumps({"processor_class": "FakeProcessor"}))
        (self.model / "model.safetensors").write_bytes(b"not-opened-as-weights")
        self.image = self.root / "frame.png"
        Image.new("RGBA", (9, 7), (10, 50, 180, 128)).save(self.image)

    def tearDown(self):
        self.temp.cleanup()

    def config(self, **changes):
        metadata = checkpoint_metadata(self.model)
        template = metadata["chat_template_candidates"][0]
        value = {
            "kind": "local_hf", "model": "reviewed-local-model", "revision": "checkpoint-r1",
            "model_path": str(self.model), "checkpoint_metadata_sha256": metadata["checkpoint_metadata_sha256"],
            "model_class": "FakeVisionModel", "processor_class": "FakeProcessor",
            "chat_template_source": template["source"], "chat_template_sha256": template["sha256"],
            "chat_message_layout": "images_then_text", "processor_call_mode": "tokenized_chat_template",
            "chat_template_kwargs": {}, "trust_remote_code": False, "local_files_only": True,
            "dtype": "bfloat16", "device": "cuda:0", "device_map": None, "input_device": "cuda:0",
            "max_memory": None, "processor_min_pixels": None, "processor_max_pixels": None,
            "generation": {"do_sample": False, "max_new_tokens": 8}, "image_order": "chronological",
            "frame_encoding": "source", "timeout_seconds": 30, "max_retries": 0,
        }
        value.update(changes)
        return value

    def test_metadata_inspection_does_not_load_weights_or_guess_template(self):
        report = inspect_local_hf(self.model)
        self.assertEqual(report["inspection_mode"], "metadata_only_no_processor_or_model_load")
        self.assertEqual(report["checkpoint"]["architectures"], ["FakeVisionModel"])
        self.assertEqual(report["checkpoint"]["chat_template_candidates"][0]["source"], "tokenizer_config.json:chat_template")
        self.assertEqual(report["checkpoint"]["weight_file_inventory"][0]["path"], "model.safetensors")
        self.assertEqual(report["chat_template_status"], "UNIQUE_CONTENT")
        (self.model / "processor_config.json").write_text(json.dumps({"chat_template": "different"}))
        self.assertEqual(inspect_local_hf(self.model)["chat_template_status"], "AMBIGUOUS_CONTENT")

    def test_processor_image_contract_probe_exercises_all_explicit_candidates_without_model_load(self):
        processor = VisualProbeProcessor()
        FakeVisionModel.last = None
        (self.model / "processor_config.json").write_text(json.dumps({"processor_class": "VisualProbeProcessor"}))
        with patch.dict(sys.modules, fake_modules(processor)):
            report = inspect_local_hf(self.model, probe_processor_images=True)
        probe = report["processor_image_contract_probe"]
        self.assertEqual(report["inspection_mode"], "metadata_plus_processor_image_contract_no_model_load")
        self.assertEqual(report["processor_probe"]["status"], "PASS")
        self.assertEqual(probe["status"], "PASS")
        self.assertFalse(probe["model_weights_loaded"])
        self.assertEqual(probe["model_from_pretrained_calls"], 0)
        self.assertEqual(probe["generate_calls"], 0)
        self.assertIsNone(FakeVisionModel.last)
        self.assertEqual(len(probe["candidates"]), 4)
        self.assertTrue(all(candidate["status"] == "PASS" for candidate in probe["candidates"]))
        self.assertTrue(all(candidate["conditions"]["grid_order_preserved_after_swap"] for candidate in probe["candidates"]))
        self.assertTrue(all(candidate["conditions"]["pixel_values_change_after_swap"] for candidate in probe["candidates"]))

    def test_processor_image_contract_probe_fails_closed_without_qwen_visual_grid(self):
        processor = MissingGridProbeProcessor()
        (self.model / "processor_config.json").write_text(json.dumps({"processor_class": "MissingGridProbeProcessor"}))
        with patch.dict(sys.modules, fake_modules(processor)):
            probe = inspect_local_hf(self.model, probe_processor_images=True)["processor_image_contract_probe"]
        self.assertEqual(probe["status"], "FAIL")
        self.assertTrue(all(candidate["status"] == "PROCESSOR_IMAGE_CONTRACT_FAILED" for candidate in probe["candidates"]))

    def test_local_config_is_closed_and_requires_all_reviewed_fields(self):
        valid = self.config()
        self.assertEqual(validate_backend(valid)["kind"], "local_hf")
        for field in ("checkpoint_metadata_sha256", "model_class", "processor_class", "chat_template_source",
                      "chat_template_sha256", "processor_call_mode", "input_device", "generation"):
            bad = dict(valid)
            bad.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_backend(bad)
        bad = dict(valid, extra={})
        with self.assertRaises(ValueError):
            validate_backend(bad)
        bad = self.config(generation={"do_sample": False})
        with self.assertRaises(ValueError):
            validate_backend(bad)
        bad = self.config(chat_template_kwargs={"messages": []})
        with self.assertRaises(ValueError):
            validate_backend(bad)

    def test_exact_model_and_template_contract_is_loaded_and_fingerprinted(self):
        processor = FakeProcessor()
        with patch.dict(sys.modules, fake_modules(processor)):
            backend = LocalHFBackend(self.config())
            before = backend.fingerprint()
            response = backend.infer({"prompt": "Use the exact prompt.", "image_paths": [str(self.image)],
                                      "frame_ids": ["f0"]})
            self.assertEqual(response, '{"status":"SUPPORTED"}')
            self.assertEqual(FakeVisionModel.last.load_kwargs["torch_dtype"], "bf16")
            self.assertTrue(FakeVisionModel.last.eval_called)
            self.assertEqual(FakeVisionModel.last.placed, "cuda:0")
            self.assertEqual(processor.load_kwargs["local_files_only"], True)
            self.assertEqual(processor.calls[0][1]["tokenize"], True)
            self.assertEqual(processor.calls[0][0][0]["content"][0]["type"], "image")
            self.assertEqual(processor.calls[0][0][0]["content"][-1], {"type": "text", "text": "Use the exact prompt."})
            self.assertEqual(FakeVisionModel.last.generate_kwargs["do_sample"], False)
            self.assertEqual(FakeVisionModel.last.generate_kwargs["max_new_tokens"], 8)
            self.assertEqual(processor.trimmed, [[3]])
            self.assertEqual(before, backend.fingerprint())
            self.assertEqual(before["preprocessing"]["decode"], "Pillow_RGB")

    def test_reviewed_template_kwargs_are_forwarded_and_audited(self):
        processor = FakeProcessor()
        with patch.dict(sys.modules, fake_modules(processor)):
            backend = LocalHFBackend(self.config(chat_template_kwargs={"enable_thinking": False}))
            backend.infer({"prompt": "Return JSON only.", "image_paths": [str(self.image)],
                           "frame_ids": ["f0"]})
            self.assertIs(processor.calls[0][1]["enable_thinking"], False)
            self.assertEqual(
                backend.fingerprint()["scientific_identity"]["chat_template_kwargs"],
                {"enable_thinking": False},
            )

    def test_template_errors_do_not_use_a_fallback_contract(self):
        processor = FakeProcessor()
        processor.raise_template_error = True
        with patch.dict(sys.modules, fake_modules(processor)):
            backend = LocalHFBackend(self.config())
            with self.assertRaisesRegex(Exception, "LOCAL_HF_TEMPLATE_APPLY_FAILURE"):
                backend.infer({"prompt": "p", "image_paths": [str(self.image)], "frame_ids": ["f0"]})
            self.assertEqual(len(processor.calls), 1)

    def test_changed_checkpoint_template_or_class_fails_closed_before_generation(self):
        processor = FakeProcessor()
        with patch.dict(sys.modules, fake_modules(processor)):
            with self.assertRaisesRegex(RuntimeError, "LOCAL_HF_CHAT_TEMPLATE_MISMATCH"):
                LocalHFBackend(self.config(chat_template_sha256="0" * 64))
            with self.assertRaisesRegex(RuntimeError, "LOCAL_HF_MODEL_CLASS_NOT_DECLARED"):
                LocalHFBackend(self.config(model_class="InventedModel"))


if __name__ == "__main__":
    unittest.main()
