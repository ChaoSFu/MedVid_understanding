from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from .checkpoint import audit_load_message, inspect_checkpoint, verify_component_paths
from .config import (
    BASELINE_NAME,
    GenerationConfig,
    OFFICIAL_SYSTEM_PROMPT,
    OFFICIAL_VTUNE_GROUNDING_PROMPT,
    PROMPT_VERSION,
    TIMESTAMP_ADAPTER_VERSION,
)
from .frame_loader import build_timechat_visual_message, build_video_features_from_frames
from .io_utils import sha256_file
from .parser import parse_timechat_answer


def static_model_fingerprint(vtune_ckpt: Path, vit_model: Path, q_former_model: Path, llama_model: Path) -> dict[str, Any]:
    return {
        "model": BASELINE_NAME,
        "vtune_ckpt": str(vtune_ckpt),
        "vtune_sha256": sha256_file(vtune_ckpt),
        "vit_model": str(vit_model),
        "vit_sha256": sha256_file(vit_model) if vit_model.is_file() else None,
        "q_former_model": str(q_former_model),
        "q_former_sha256": sha256_file(q_former_model) if q_former_model.is_file() else None,
        "llama_model": str(llama_model),
        "max_frame_pos": 96,
        "window_size": 32,
        "stride": 32,
        "qformer_text_input": True,
        "lora": True,
        "lora_inference_mode": True,
        "prompt_version": PROMPT_VERSION,
        "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
    }


def _set_cfg_value(cfg: Any, key: str, value: str) -> None:
    if hasattr(cfg, key):
        setattr(cfg, key, value)
    elif hasattr(cfg, "__setitem__"):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def install_transformers_compat_shims() -> dict[str, str]:
    """Patch old TimeChat/LAVIS import locations for newer Transformers builds."""
    patched: dict[str, str] = {}
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except Exception as exc:
        return {"shim_error": repr(exc)}

    for name in ("apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
            patched[name] = "transformers.pytorch_utils"
    return patched


def install_optional_dependency_shims() -> dict[str, str]:
    """Provide no-op modules for optional training/logging deps unused in inference."""
    patched: dict[str, str] = {}
    if "wandb" not in sys.modules:
        wandb = ModuleType("wandb")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb
        patched["wandb"] = "noop_logging_module"
    return patched


def patch_generation_mixin_methods(model: Any) -> dict[str, Any]:
    """Restore `.generate()` for old model classes under newer Transformers."""
    try:
        from transformers import GenerationConfig as TransformersGenerationConfig
        from transformers.generation.utils import GenerationMixin
    except Exception as exc:
        return {"shim_error": [repr(exc)]}

    patched_classes: dict[type, list[str]] = {}
    patched_instances: dict[str, list[str]] = {}
    original_validate = GenerationMixin._validate_model_kwargs

    def validate_model_kwargs_allow_inputs_embeds(self, model_kwargs):
        filtered = dict(model_kwargs)
        filtered.pop("inputs_embeds", None)
        return original_validate(self, filtered)

    def usable_past_key_values(value):
        if value is None:
            return None
        try:
            first_layer = value[0]
            first_key = first_layer[0]
        except Exception:
            return value
        return value if hasattr(first_key, "shape") else None

    def prepare_inputs_for_generation_allow_inputs_embeds(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        past_key_values = usable_past_key_values(past_key_values)
        base_model = getattr(self, "model", None) or getattr(self, "base_model", None)
        target = getattr(base_model, "prepare_inputs_for_generation", None)
        if callable(target):
            try:
                prepared = target(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )
                if isinstance(prepared, dict):
                    prepared_past = usable_past_key_values(prepared.get("past_key_values"))
                    if prepared_past is None:
                        prepared.pop("past_key_values", None)
                    else:
                        prepared["past_key_values"] = prepared_past
                return prepared
            except TypeError:
                pass
        model_inputs = dict(kwargs)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        return model_inputs

    modules = model.modules() if hasattr(model, "modules") else [model]
    for module in modules:
        if not hasattr(module, "prepare_inputs_for_generation"):
            continue
        instance_changes: list[str] = []
        if getattr(module, "generation_config", None) is None:
            model_config = getattr(module, "config", None)
            try:
                module.generation_config = TransformersGenerationConfig.from_model_config(model_config) if model_config else TransformersGenerationConfig()
            except Exception:
                module.generation_config = TransformersGenerationConfig()
            instance_changes.append("generation_config")
        if instance_changes:
            patched_instances[f"{type(module).__module__}.{type(module).__name__}@{id(module)}"] = instance_changes
        cls = type(module)
        if cls in patched_classes:
            continue
        added: list[str] = []
        try:
            prepare_signature = inspect.signature(module.prepare_inputs_for_generation)
        except (AttributeError, TypeError, ValueError):
            prepare_signature = None
        if prepare_signature is None or "inputs_embeds" not in prepare_signature.parameters:
            cls.prepare_inputs_for_generation = prepare_inputs_for_generation_allow_inputs_embeds
            added.append("prepare_inputs_for_generation_allow_inputs_embeds")
        if not hasattr(cls, "_is_stateful"):
            cls._is_stateful = False
            added.append("_is_stateful_false")
        cls._validate_model_kwargs = validate_model_kwargs_allow_inputs_embeds
        added.append("_validate_model_kwargs_allow_inputs_embeds")
        if not hasattr(module, "generate"):
            for name, value in GenerationMixin.__dict__.items():
                if name.startswith("__") or hasattr(cls, name):
                    continue
                setattr(cls, name, value)
                added.append(name)
        patched_classes[cls] = added
    return {
        "classes": {f"{cls.__module__}.{cls.__name__}": names for cls, names in patched_classes.items()},
        "instances": patched_instances,
    }


@dataclass
class TimeChatLoadResult:
    runner: "MedVidUTimeChatVTune"
    checkpoint_inspection: dict[str, Any]
    checkpoint_load_audit: dict[str, Any]
    model_fingerprint: dict[str, Any]


class _LoadStateDictCapture:
    def __init__(self, checkpoint_inspection: dict[str, Any] | None = None):
        self.matches: list[dict[str, Any]] = []
        self._original = None
        self.checkpoint_inspection = checkpoint_inspection

    def __enter__(self):
        import torch

        self._original = torch.nn.Module.load_state_dict
        capture = self

        def wrapped(module, state_dict, strict=True, assign=False):
            if assign is False:
                msg = capture._original(module, state_dict, strict=strict)
            else:
                try:
                    msg = capture._original(module, state_dict, strict=strict, assign=assign)
                except TypeError:
                    msg = capture._original(module, state_dict, strict=strict)
            keys = list(state_dict.keys()) if hasattr(state_dict, "keys") else []
            if "video_frame_position_embedding.weight" in keys or any(k.startswith("video_Qformer.") for k in keys):
                payload = audit_load_message(msg, checkpoint_inspection=capture.checkpoint_inspection)
                payload["module_class"] = type(module).__name__
                payload["state_dict_keys"] = len(keys)
                capture.matches.append(payload)
            return msg

        torch.nn.Module.load_state_dict = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        import torch

        if self._original is not None:
            torch.nn.Module.load_state_dict = self._original


class MedVidUTimeChatVTune:
    def __init__(self, timechat_repo: Path, model: Any, vis_processor: Any, chat: Any, args: Any, model_config: Any):
        self.timechat_repo = timechat_repo
        self.model = model
        self.vis_processor = vis_processor
        self.chat = chat
        self.args = args
        self.model_config = model_config
        self.debug = bool(getattr(args, "debug", False))
        self.n_frames = 96
        self.CoT = False

    @classmethod
    def load(
        cls,
        timechat_repo: Path,
        vtune_ckpt: Path,
        vit_model: Path,
        q_former_model: Path,
        llama_model: Path,
        gpu_id: int = 0,
    ) -> TimeChatLoadResult:
        verify_component_paths(vtune_ckpt, vit_model, q_former_model, llama_model)
        checkpoint_inspection = inspect_checkpoint(vtune_ckpt)
        if str(timechat_repo) not in sys.path:
            sys.path.insert(0, str(timechat_repo))

        compat_shims = install_transformers_compat_shims()
        optional_shims = install_optional_dependency_shims()
        from timechat.common.config import Config
        from timechat.common.registry import registry
        from timechat.conversation.conversation_video import Chat
        from utils.prompts import prompt

        args = SimpleNamespace(
            cfg_path=str(timechat_repo / "timechat" / "eval_configs" / "timechat.yaml"),
            gpu_id=gpu_id,
            options=None,
            dset_name="medvidu",
            task="grounding",
            fine_tuned=True,
            debug=False,
            CoT=False,
        )
        cfg = Config(args)
        _set_cfg_value(cfg.model_cfg, "ckpt", str(vtune_ckpt))
        _set_cfg_value(cfg.model_cfg, "activitynet_ckpt", str(vtune_ckpt))
        _set_cfg_value(cfg.model_cfg, "vit_model", str(vit_model))
        _set_cfg_value(cfg.model_cfg, "q_former_model", str(q_former_model))
        _set_cfg_value(cfg.model_cfg, "llama_model", str(llama_model))
        _set_cfg_value(cfg.model_cfg, "device_8bit", gpu_id)
        prompt["grounding"] = OFFICIAL_VTUNE_GROUNDING_PROMPT

        model_cls = registry.get_model_class(cfg.model_cfg.arch)
        with _LoadStateDictCapture(checkpoint_inspection=checkpoint_inspection) as capture:
            model = model_cls.from_config(cfg.model_cfg).to(f"cuda:{gpu_id}")
        model.eval()
        generation_mixin_shims = patch_generation_mixin_methods(model)
        if not capture.matches:
            raise RuntimeError("STOP: VTune checkpoint load audit was not captured; refusing to continue.")
        checkpoint_load_audit = capture.matches[-1]
        checkpoint_load_audit["transformers_compat_shims"] = compat_shims
        checkpoint_load_audit["optional_dependency_shims"] = optional_shims
        checkpoint_load_audit["generation_mixin_shims"] = generation_mixin_shims
        if checkpoint_load_audit.get("critical_mismatch"):
            raise RuntimeError("STOP: critical TimeChat checkpoint keys are missing; see checkpoint_load_audit.json")

        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        chat = Chat(model, vis_processor, device=f"cuda:{gpu_id}")
        runner = cls(timechat_repo=timechat_repo, model=model, vis_processor=vis_processor, chat=chat, args=args, model_config=cfg.model_cfg)
        fingerprint = runner.model_fingerprint(vtune_ckpt, vit_model, q_former_model, llama_model)
        return TimeChatLoadResult(runner, checkpoint_inspection, checkpoint_load_audit, fingerprint)

    def model_fingerprint(self, vtune_ckpt: Path, vit_model: Path, q_former_model: Path, llama_model: Path) -> dict[str, Any]:
        static = static_model_fingerprint(vtune_ckpt, vit_model, q_former_model, llama_model)
        cfg = self.model_config
        payload = {
            **static,
            "max_frame_pos": int(cfg.get("max_frame_pos", 96)),
            "window_size": int(cfg.get("window_size", 32)),
            "stride": int(cfg.get("stride", 32)),
            "qformer_text_input": bool(cfg.get("qformer_text_input", True)),
            "lora": bool(cfg.get("lora", True)),
            "lora_inference_mode": bool(cfg.get("lora_inference_mode", True)),
        }
        return payload

    def initialize_chat(self, task: str, msg: str, add_detail: str | None = None):
        from timechat.conversation.conversation_video import conv_llava_llama_2

        chat_state = conv_llava_llama_2.copy()
        chat_state.system = OFFICIAL_SYSTEM_PROMPT
        if add_detail:
            chat_state.system += " " + add_detail
        return chat_state

    def inference(self, chat_state: Any, video_features: list[Any], generation: GenerationConfig) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        chat_state.append_message(chat_state.roles[1], None)
        embs = self.chat.get_context_emb(chat_state, video_features)
        current_max_len = embs.shape[1] + generation.max_new_tokens
        if current_max_len - generation.max_length > 0:
            begin_idx = max(0, current_max_len - generation.max_length)
            embs = embs[:, begin_idx:]

        tokenizer = self.model.llama_tokenizer
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        dummy_input_ids = torch.full(
            (embs.shape[0], embs.shape[1]),
            int(pad_token_id),
            dtype=torch.long,
            device=embs.device,
        )
        attention_mask = torch.ones(dummy_input_ids.shape, dtype=torch.long, device=embs.device)
        stopping_criteria = getattr(self.chat, "stopping_criteria", None)
        if stopping_criteria is None:
            stop_words_ids = [torch.tensor([2], device=embs.device)]

            class StoppingCriteriaSub(StoppingCriteria):
                def __init__(self, stops):
                    super().__init__()
                    self.stops = stops

                def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                    for stop in self.stops:
                        if torch.all((stop == input_ids[0][-len(stop) :])).item():
                            return True
                    return False

            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        outputs = self.model.llama_model.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=embs,
            attention_mask=attention_mask,
            max_new_tokens=generation.max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=generation.num_beams,
            do_sample=generation.do_sample,
            temperature=generation.temperature,
            max_length=generation.max_length,
            pad_token_id=pad_token_id,
        )
        output_token = outputs[0]
        if output_token.shape[0] > embs.shape[1]:
            output_token = output_token[embs.shape[1] :]
        if output_token.shape[0] > 0 and output_token[0].item() in {0, 1}:
            output_token = output_token[1:]
        if output_token.shape[0] > 0 and output_token[0].item() == int(pad_token_id):
            output_token = output_token[1:]
        llm_message = tokenizer.decode(output_token, add_special_tokens=False)
        llm_message = llm_message.split("###")[0]
        llm_message = llm_message.split("Assistant:")[-1].strip()
        chat_state.messages[-1][-1] = llm_message
        return str(llm_message)

    def extract_time(self, paragraph: str):
        import re

        paragraph = (paragraph or "").lower()
        sentences = re.split(r"[!?\n]", paragraph)
        keywords = ["starts", "ends", "happens in", "start time", "end time", "start", "end", "happen"]
        candidates = [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]
        timestamps = []
        for time_pattern in [r"(\d+\.*\d*)\s*-\s*(\d+\.*\d*)"]:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [[float(start), float(end)] for start, end in time_matches]
        if len(timestamps) == 0:
            times = []
            time_regex = re.compile(r"\b(\d+\.\d+\b|\b\d+)\b")
            for sentence in candidates:
                time = re.findall(time_regex, sentence)
                if time:
                    times.append(float(time[0]))
            times = times[: len(times) // 2 * 2]
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
        if len(timestamps) == 0:
            times = []
            time_regex = re.compile(r"\b((\d{1,2}:\d{2}:\d{2}))\b")
            for sentence in candidates:
                time = re.findall(time_regex, sentence)
                if not time:
                    continue
                t = time[0][0]
                h, m, s = map(int, t.split(":"))
                times.append(h * 3600 + m * 60 + s)
            times = times[: len(times) // 2 * 2]
            timestamps = [(times[i], times[i + 1]) for i in range(0, len(times), 2)]
        results = [[min(start, end), max(start, end)] for start, end in timestamps]
        return results[0] if results else [0, 0]

    def extract_time2(self, paragraph: str):
        import re

        pattern = r"(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)\s*(?:to|and|–|—)\s*(?:(?:from\s+|in\s+|between\s+|at\s+|—|to\s+|and\s+|–)\s*)?(\d+(?:\.\d+)?)?\s*seconds?"
        match = re.search(pattern, paragraph or "")
        if match:
            start_timestamp = float(match.group(1))
            end_timestamp = float(match.group(2)) if match.group(2) is not None else None
            return [0, start_timestamp] if end_timestamp is None else [start_timestamp, end_timestamp]
        matches = re.findall(r"\d+\.\d+|\d+", paragraph or "")
        float_numbers = [float(match) for match in matches]
        if len(float_numbers) == 0:
            return [0, 0]
        if len(float_numbers) == 1:
            return [0, float_numbers[0]]
        return float_numbers[:2]

    def run_one(self, row: dict[str, Any], generation: GenerationConfig) -> dict[str, Any]:
        frame_paths = list(row["ordered_frame_paths"])
        local_timestamps = [float(t) for t in row["local_timestamps"]]
        video_features, msg, selected = build_video_features_from_frames(
            self.model,
            self.vis_processor,
            frame_paths,
            local_timestamps,
            device=f"cuda:{self.args.gpu_id}",
        )
        question = OFFICIAL_VTUNE_GROUNDING_PROMPT.format(event=row["human_question"])
        chat_state = self.initialize_chat("grounding", msg)
        chat_state.append_message(chat_state.roles[0], build_timechat_visual_message(msg))
        self.chat.ask(question, chat_state)
        answer = self.inference(chat_state, video_features, generation)
        parsed = parse_timechat_answer(self, answer)
        return {
            **parsed,
            "question_sent_to_model": question,
            "timechat_msg": msg,
            "n_selected_frames": selected["selection_audit"]["n_selected_frames"],
            "selected_logical_indices": selected["selection_audit"]["selected_logical_indices"],
            "selected_local_timestamps": selected["selected_local_timestamps"],
            "selected_rounded_timestamps": selected["selected_rounded_timestamps"],
            "timestamp_texts": selected["timestamp_texts"],
            "video_tensor_shape": selected.get("video_tensor_shape"),
            "video_embedding_shape": selected.get("video_embedding_shape"),
        }
