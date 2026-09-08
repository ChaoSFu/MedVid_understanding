import unittest

from evidence_stability.models.qwen3_vl import Qwen3VLVideoWindowModel, _model_identity_hash, _parse_max_memory


class _ModelStub:
    def __init__(self, device_map):
        self.hf_device_map = device_map


class LocalHFDeviceMapTests(unittest.TestCase):
    def test_sharded_model_uses_first_cuda_shard_for_inputs(self):
        model = Qwen3VLVideoWindowModel.__new__(Qwen3VLVideoWindowModel)
        model.device = "cuda:0"
        model.device_map = "balanced"
        model.model = _ModelStub({"visual": 0, "language_model.layers.0": 1})
        self.assertEqual(model._input_device(), "cuda:0")

    def test_sharded_model_accepts_string_cuda_device(self):
        model = Qwen3VLVideoWindowModel.__new__(Qwen3VLVideoWindowModel)
        model.device = "cuda:0"
        model.device_map = "balanced"
        model.model = _ModelStub({"visual": "cuda:1"})
        self.assertEqual(model._input_device(), "cuda:1")

    def test_single_gpu_model_uses_requested_device(self):
        model = Qwen3VLVideoWindowModel.__new__(Qwen3VLVideoWindowModel)
        model.device = "cuda:0"
        model.device_map = None
        self.assertEqual(model._input_device(), "cuda:0")

    def test_max_memory_parser_uses_visible_cuda_indices(self):
        self.assertEqual(_parse_max_memory("cuda:0=35GiB,cuda:1=70GiB"), {0: "35GiB", 1: "70GiB"})

    def test_max_memory_parser_rejects_malformed_entry(self):
        with self.assertRaises(ValueError):
            _parse_max_memory("cuda:0")

    def test_runtime_placement_does_not_change_model_identity_hash(self):
        base = {
            "model_path": "/models/Qwen3.5-9B",
            "config_sha256": "config",
            "generation_config_sha256": "generation",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "processor_class": "Qwen3VLProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "transformers_version": "5.16.1",
            "torch_version": "2.6.0+cu124",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
            "processor_min_pixels": None,
            "processor_max_pixels": None,
        }
        single_gpu = {**base, "device_map": None, "max_memory": None}
        sharded = {**base, "device_map": "balanced", "max_memory": {0: "35GiB", 1: "70GiB"}}
        self.assertEqual(_model_identity_hash(single_gpu), _model_identity_hash(sharded))
