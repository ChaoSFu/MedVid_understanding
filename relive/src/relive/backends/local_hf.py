"""Strict local Hugging Face backend for an already-reviewed checkpoint.

This adapter intentionally has no Qwen-specific class list, prompt template, or
processor fallback.  A metadata-only ``inspect-local-hf`` report must first be
used to populate the exact architecture, template hash, call mode, and input
placement in the configuration.  The adapter then fails closed if those
declarations no longer match the local checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image

from relive.config import validate_backend
from .base import Backend, BackendError
from .inspection import CheckpointInspectionError, checkpoint_metadata


ADAPTER_VERSION = "relive-local-hf-v1"


def _identity_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _torch_dtype(torch_module: Any, name: str) -> Any:
    return {"bfloat16": torch_module.bfloat16, "float16": torch_module.float16,
            "float32": torch_module.float32}[name]


def _max_memory(value: dict[str, str] | None) -> dict[int | str, str] | None:
    if value is None:
        return None
    parsed: dict[int | str, str] = {}
    for key, limit in value.items():
        if key.startswith("cuda:"):
            parsed[int(key.removeprefix("cuda:"))] = limit
        elif key.isdigit():
            parsed[int(key)] = limit
        else:
            parsed[key] = limit
    return parsed


def _processor_template(processor: Any) -> str | None:
    template = getattr(processor, "chat_template", None)
    if isinstance(template, str):
        return template
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    return template if isinstance(template, str) else None


class LocalHFBackend(Backend):
    """One exact Hugging Face model and one exact checkpoint chat contract."""

    synthetic = False

    def __init__(self, config: dict[str, Any]):
        validated = validate_backend(config)
        if validated["kind"] != "local_hf":
            raise ValueError("LocalHFBackend requires kind=local_hf")
        super().__init__(validated)
        try:
            self.checkpoint = checkpoint_metadata(self.config["model_path"])
        except CheckpointInspectionError as exc:
            raise RuntimeError("LOCAL_HF_CHECKPOINT_METADATA_FAILURE") from exc
        self._validate_declared_checkpoint()
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("LOCAL_HF_DEPENDENCY_UNAVAILABLE") from exc
        self.torch = torch
        self.transformers = transformers
        self._load_exact_checkpoint()

    def _validate_declared_checkpoint(self) -> None:
        if self.checkpoint["checkpoint_metadata_sha256"] != self.config["checkpoint_metadata_sha256"]:
            raise RuntimeError("LOCAL_HF_CHECKPOINT_METADATA_MISMATCH")
        architectures = self.checkpoint["architectures"]
        if self.config["model_class"] not in architectures:
            raise RuntimeError("LOCAL_HF_MODEL_CLASS_NOT_DECLARED")
        matched = [candidate for candidate in self.checkpoint["chat_template_candidates"]
                   if candidate["source"] == self.config["chat_template_source"]
                   and candidate["sha256"] == self.config["chat_template_sha256"]]
        if len(matched) != 1:
            raise RuntimeError("LOCAL_HF_CHAT_TEMPLATE_MISMATCH")

    def _load_exact_checkpoint(self) -> None:
        model_class = getattr(self.transformers, self.config["model_class"], None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            raise RuntimeError("LOCAL_HF_UNSUPPORTED_DECLARED_ARCHITECTURE")
        auto_processor = getattr(self.transformers, "AutoProcessor", None)
        if auto_processor is None or not hasattr(auto_processor, "from_pretrained"):
            raise RuntimeError("LOCAL_HF_AUTOPROCESSOR_UNAVAILABLE")
        processor_kwargs = {
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": True,
        }
        for config_key, processor_key in (("processor_min_pixels", "min_pixels"),
                                          ("processor_max_pixels", "max_pixels")):
            if self.config[config_key] is not None:
                processor_kwargs[processor_key] = self.config[config_key]
        try:
            self.processor = auto_processor.from_pretrained(self.config["model_path"], **processor_kwargs)
        except Exception as exc:
            raise RuntimeError("LOCAL_HF_PROCESSOR_LOAD_FAILURE") from exc
        if self.processor.__class__.__name__ != self.config["processor_class"]:
            raise RuntimeError("LOCAL_HF_PROCESSOR_CLASS_MISMATCH")
        actual_template = _processor_template(self.processor)
        if actual_template is None or hashlib.sha256(actual_template.encode("utf-8")).hexdigest() != self.config["chat_template_sha256"]:
            raise RuntimeError("LOCAL_HF_LOADED_TEMPLATE_MISMATCH")
        model_kwargs = {
            "torch_dtype": _torch_dtype(self.torch, self.config["dtype"]),
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": True,
        }
        if self.config["device_map"] is not None:
            model_kwargs["device_map"] = self.config["device_map"]
            max_memory = _max_memory(self.config["max_memory"])
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
        try:
            self.model = model_class.from_pretrained(self.config["model_path"], **model_kwargs)
        except Exception as exc:
            raise RuntimeError("LOCAL_HF_MODEL_LOAD_FAILURE") from exc
        if self.model.__class__.__name__ != self.config["model_class"]:
            raise RuntimeError("LOCAL_HF_LOADED_MODEL_CLASS_MISMATCH")
        if self.config["device_map"] is None:
            try:
                self.model.to(self.config["device"])
            except Exception as exc:
                raise RuntimeError("LOCAL_HF_MODEL_PLACEMENT_FAILURE") from exc
        self.model.eval()
        self._fingerprint = self._make_fingerprint()

    def _make_fingerprint(self) -> dict[str, Any]:
        placement = {
            "dtype": self.config["dtype"], "device": self.config["device"],
            "device_map": self.config["device_map"], "input_device": self.config["input_device"],
            "max_memory": self.config["max_memory"],
        }
        scientific_identity = {
            "adapter_version": ADAPTER_VERSION,
            "model": self.config["model"], "model_path": self.config["model_path"],
            "revision": self.config["revision"],
            "checkpoint_metadata_sha256": self.checkpoint["checkpoint_metadata_sha256"],
            "model_class": self.config["model_class"], "processor_class": self.config["processor_class"],
            "chat_template_source": self.config["chat_template_source"],
            "chat_template_sha256": self.config["chat_template_sha256"],
            "chat_message_layout": self.config["chat_message_layout"],
            "processor_call_mode": self.config["processor_call_mode"],
            "chat_template_kwargs": self.config["chat_template_kwargs"],
            "trust_remote_code": self.config["trust_remote_code"], "local_files_only": True,
            "processor_min_pixels": self.config["processor_min_pixels"],
            "processor_max_pixels": self.config["processor_max_pixels"],
            "generation": self.config["generation"], "image_order": self.config["image_order"],
            "frame_encoding": self.config["frame_encoding"],
            "jpeg_quality": self.config.get("jpeg_quality"),
            "transformers_version": getattr(self.transformers, "__version__", None),
            "torch_version": getattr(self.torch, "__version__", None),
            "actual_model_class": self.model.__class__.__name__,
            "actual_processor_class": self.processor.__class__.__name__,
        }
        return {
            "adapter_version": ADAPTER_VERSION, "synthetic": False, "scientific_identity": scientific_identity,
            "model_identity_hash": _identity_hash(scientific_identity), "runtime_placement": placement,
            "checkpoint_metadata": deepcopy(self.checkpoint),
            "timeout_seconds": self.config["timeout_seconds"],
            "timeout_enforcement": "NOT_SUPPORTED_FOR_IN_PROCESS_GENERATE",
            "max_retries": self.config["max_retries"],
            "preprocessing": {"decode": "Pillow_RGB", "frame_encoding": self.config["frame_encoding"],
                              "jpeg_quality": self.config.get("jpeg_quality"), "resize": False},
        }

    def fingerprint(self) -> dict[str, Any]:
        return deepcopy(self._fingerprint)

    def _prepared_image(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB").copy()
        except (OSError, ValueError):
            raise BackendError("LOCAL_HF_IMAGE_DECODE_FAILURE") from None
        encoding = self.config["frame_encoding"]
        if encoding == "source":
            return image
        buffer = BytesIO()
        try:
            options = {"quality": self.config["jpeg_quality"]} if encoding == "jpeg" else {}
            image.save(buffer, format=encoding.upper(), **options)
            with Image.open(BytesIO(buffer.getvalue())) as reloaded:
                return reloaded.convert("RGB").copy()
        except (OSError, ValueError):
            raise BackendError("LOCAL_HF_IMAGE_ENCODING_FAILURE") from None

    def _messages(self, request: dict[str, Any], images: list[Image.Image]) -> list[dict[str, Any]]:
        paths = request.get("image_paths", [])
        frame_ids = request.get("frame_ids", [])
        if not isinstance(paths, list) or not isinstance(frame_ids, list) or len(paths) != len(frame_ids):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        if self.config["chat_message_layout"] == "images_then_text":
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": request["prompt"]})
        else:
            content = [{"type": "text", "text": request["prompt"]}]
            content.extend({"type": "image", "image": image} for image in images)
        return [{"role": "user", "content": content}]

    def _model_inputs(self, messages: list[dict[str, Any]], images: list[Image.Image]) -> Any:
        kwargs = dict(self.config["chat_template_kwargs"])
        try:
            if self.config["processor_call_mode"] == "tokenized_chat_template":
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt", **kwargs
                )
            else:
                rendered = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
                if not isinstance(rendered, str):
                    raise TypeError("checkpoint template did not render text")
                inputs = self.processor(text=[rendered], images=images, return_tensors="pt")
        except Exception:
            raise BackendError("LOCAL_HF_TEMPLATE_APPLY_FAILURE") from None
        if not isinstance(inputs, dict) and not hasattr(inputs, "items"):
            raise BackendError("LOCAL_HF_PROCESSOR_FAILURE")
        if "input_ids" not in inputs:
            raise BackendError("LOCAL_HF_PROCESSOR_FAILURE")
        try:
            return inputs.to(self.config["input_device"]) if hasattr(inputs, "to") else {
                key: value.to(self.config["input_device"]) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        except Exception:
            raise BackendError("LOCAL_HF_INPUT_PLACEMENT_FAILURE") from None

    def infer(self, request: dict[str, Any]) -> str:
        self.calls += 1
        paths = request.get("image_paths", [])
        if not isinstance(paths, list):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        try:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **self.config["generation"])
            input_ids = inputs["input_ids"]
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ValueError("invalid decode")
            return decoded[0].strip()
        except BackendError:
            raise
        except Exception:
            raise BackendError("LOCAL_HF_GENERATION_FAILURE") from None

    def infer_with_generation_metadata(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate once and expose non-semantic stopping metadata.

        This is deliberately separate from :meth:`infer`, so existing verifier
        calls retain their exact generation behavior and return type.  It does
        not attempt grammar/schema constrained decoding: availability of the
        native ``generate`` metadata is reported by callers independently from
        any claim that the checkpoint can enforce a JSON schema.
        """
        self.calls += 1
        paths = request.get("image_paths", [])
        if not isinstance(paths, list):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        generation = dict(self.config["generation"])
        generation["return_dict_in_generate"] = True
        try:
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            sequences = getattr(output, "sequences", None)
            if sequences is None:
                raise ValueError("generate did not return sequences")
            input_ids = inputs["input_ids"]
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, sequences)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ValueError("invalid decode")
            token_ids = trimmed[0].detach().to("cpu").tolist() if hasattr(trimmed[0], "detach") else list(trimmed[0])
            if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
                raise ValueError("invalid generated token ids")
            configured_max = generation.get("max_new_tokens")
            if type(configured_max) is not int or configured_max <= 0:
                raise ValueError("missing max_new_tokens")
            eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
            eos_ids = {eos} if type(eos) is int else set(eos or []) if isinstance(eos, (list, tuple, set)) else set()
            finish_reason = (
                "EOS_TOKEN" if token_ids and token_ids[-1] in eos_ids
                else "MAX_NEW_TOKENS" if len(token_ids) >= configured_max
                else "OTHER_STOP"
            )
            return {
                "raw_response": decoded[0].strip(),
                "generation_metadata": {
                    "finish_reason": finish_reason,
                    "generated_token_count": len(token_ids),
                    "max_new_tokens": configured_max,
                    "reached_max_new_tokens": len(token_ids) >= configured_max,
                },
            }
        except BackendError:
            raise
        except Exception:
            raise BackendError("LOCAL_HF_GENERATION_FAILURE") from None

    def audit_token_constraint(self, constraint: Any) -> dict[str, Any]:
        """Bind a token grammar to the loaded checkpoint tokenizer without generation."""
        if not callable(getattr(constraint, "prepare", None)):
            raise BackendError("TOKEN_CONSTRAINT_REQUEST_INVALID")
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_ids = {eos} if type(eos) is int else set(eos or []) if isinstance(eos, (list, tuple, set)) else set()
        try:
            binding = constraint.prepare(tokenizer, 0, eos_ids)
            initial_allowed = constraint.allowed_token_ids([])
            if not initial_allowed:
                raise ValueError("no legal initial grammar token")
            get_vocab = getattr(tokenizer, "get_vocab", None)
            vocabulary = get_vocab() if callable(get_vocab) else None
            template = _processor_template(self.processor)
            if not isinstance(vocabulary, dict):
                raise ValueError("tokenizer vocabulary unavailable")
            return {"binding": binding, "initial_allowed_token_count": len(initial_allowed),
                    "tokenizer_binding": {"tokenizer_class": tokenizer.__class__.__name__,
                     "tokenizer_vocab_size": len(vocabulary),
                     "tokenizer_vocabulary_sha256": _identity_hash(vocabulary),
                     "tokenizer_chat_template_sha256": hashlib.sha256((template or "").encode("utf-8")).hexdigest(),
                     "eos_token_ids": sorted(eos_ids)}}
        except BackendError:
            raise
        except Exception:
            raise BackendError("LOCAL_HF_TOKEN_CONSTRAINT_AUDIT_FAILURE") from None

    def infer_with_token_constraint(self, request: dict[str, Any], constraint: Any) -> dict[str, Any]:
        """Run the actual multimodal ``generate`` path with a fail-closed token constraint.

        ``constraint`` is a versioned object supplied by a diagnostic caller.  It
        receives the real tokenizer and prompt token count, masks logits for each
        generated token, and returns explicit metadata.  This is separate from
        both existing inference methods so their behavior cannot drift.
        """
        self.calls += 1
        paths = request.get("image_paths", [])
        if not isinstance(paths, list) or not callable(getattr(constraint, "prepare", None)):
            raise BackendError("TOKEN_CONSTRAINT_REQUEST_INVALID")
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        tokenizer = getattr(self.processor, "tokenizer", None)
        input_ids = inputs.get("input_ids")
        if tokenizer is None or input_ids is None or getattr(input_ids, "shape", None) is None:
            raise BackendError("TOKEN_CONSTRAINT_TOKENIZER_UNAVAILABLE")
        if len(input_ids.shape) != 2 or int(input_ids.shape[0]) != 1:
            raise BackendError("TOKEN_CONSTRAINT_BATCH_UNSUPPORTED")
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        eos_ids = {eos} if type(eos) is int else set(eos or []) if isinstance(eos, (list, tuple, set)) else set()
        if not eos_ids:
            raise BackendError("TOKEN_CONSTRAINT_EOS_UNAVAILABLE")
        try:
            binding = constraint.prepare(tokenizer, int(input_ids.shape[1]), eos_ids)
            get_vocab = getattr(tokenizer, "get_vocab", None)
            vocabulary = get_vocab() if callable(get_vocab) else None
            if not isinstance(vocabulary, dict) or any(not isinstance(k, str) or type(v) is not int for k, v in vocabulary.items()):
                raise ValueError("tokenizer vocabulary unavailable")
            template = _processor_template(self.processor)
            tokenizer_binding = {"tokenizer_class": tokenizer.__class__.__name__,
                "tokenizer_vocab_size": len(vocabulary),
                "tokenizer_vocabulary_sha256": _identity_hash(vocabulary),
                "tokenizer_chat_template_sha256": hashlib.sha256((template or "").encode("utf-8")).hexdigest(),
                "prompt_token_count": int(input_ids.shape[1]), "eos_token_ids": sorted(eos_ids)}
            transformers = self.transformers
            base = getattr(transformers, "LogitsProcessor", object)
            outer = constraint
            class _ConstrainedProcessor(base):
                def __call__(self, ids: Any, scores: Any) -> Any:
                    return outer.apply_logits(ids, scores)
            processor_list = getattr(transformers, "LogitsProcessorList", None)
            if processor_list is None:
                raise ValueError("transformers logits processor API unavailable")
            generation = dict(self.config["generation"])
            generation["return_dict_in_generate"] = True
            generation["logits_processor"] = processor_list([_ConstrainedProcessor()])
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            sequences = getattr(output, "sequences", None)
            if sequences is None:
                raise ValueError("generate did not return sequences")
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, sequences)]
            token_ids = trimmed[0].detach().to("cpu").tolist() if hasattr(trimmed[0], "detach") else list(trimmed[0])
            if not isinstance(token_ids, list) or any(type(token) is not int for token in token_ids):
                raise ValueError("invalid generated token ids")
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ValueError("invalid decode")
            configured_max = generation.get("max_new_tokens")
            finish_reason = ("EOS_TOKEN" if token_ids and token_ids[-1] in eos_ids
                             else "MAX_NEW_TOKENS" if len(token_ids) >= configured_max else "OTHER_STOP")
            return {"raw_response": decoded[0].strip(),
                    "generation_metadata": {"finish_reason": finish_reason,
                        "generated_token_count": len(token_ids),
                        "max_new_tokens": configured_max,
                        "reached_max_new_tokens": len(token_ids) >= configured_max},
                    "constraint_metadata": {"binding": binding, "tokenizer_binding": tokenizer_binding,
                        "execution": constraint.final_metadata(token_ids),
                        "generated_token_ids_sha256": hashlib.sha256(
                            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()}}
        except BackendError:
            raise
        except Exception:
            raise BackendError("LOCAL_HF_TOKEN_CONSTRAINED_GENERATION_FAILURE") from None

    def forced_choice_token_contract(self, request: dict[str, Any], choices: tuple[str, ...] = ("A", "B", "C")) -> dict[str, Any]:
        """Validate exact one-token diagnostic choices in the real chat context.

        This is intentionally separate from :meth:`infer`: it neither calls
        ``generate`` nor parses a semantic response.  The prompt is required to
        end in the fixed answer anchor so a next-token A/B/C comparison has an
        unambiguous meaning.
        """
        if not isinstance(request.get("prompt"), str) or not request["prompt"].endswith("\nAnswer:"):
            raise BackendError("CHOICE_PROMPT_MISSING_ANSWER_ANCHOR")
        if (not isinstance(choices, tuple) or not choices or len(set(choices)) != len(choices)
                or any(not isinstance(label, str) or len(label) != 1 or not label.isascii() or not label.isupper()
                       for label in choices)):
            raise BackendError("CHOICE_LABEL_CONTRACT_INVALID")
        paths = request.get("image_paths", [])
        if not isinstance(paths, list):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        tokenizer = getattr(self.processor, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise BackendError("CHOICE_TOKENIZER_UNAVAILABLE")
        token_ids: dict[str, int] = {}
        try:
            for label in choices:
                encoded = encode(label, add_special_tokens=False)
                if not isinstance(encoded, list) or len(encoded) != 1 or type(encoded[0]) is not int:
                    raise ValueError("choice does not map to exactly one token")
                token_ids[label] = encoded[0]
        except Exception:
            raise BackendError("CHOICE_TOKENIZATION_NOT_UNIQUE") from None
        if len(set(token_ids.values())) != len(choices):
            raise BackendError("CHOICE_TOKENIZATION_NOT_UNIQUE")
        try:
            input_ids = inputs["input_ids"]
            sequence = input_ids[0].detach().to("cpu").tolist()
            if not isinstance(sequence, list) or not sequence:
                raise ValueError("empty chat context")
        except Exception:
            raise BackendError("CHOICE_CONTEXT_TOKENIZATION_UNAVAILABLE") from None
        return {"choice_labels": list(choices), "choice_token_ids": token_ids,
                "context_input_ids_sha256": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest(),
                "context_token_count": len(sequence)}

    def forced_choice_likelihood(self, request: dict[str, Any], *, choice_token_ids: dict[str, int] | None = None,
                                 choices: tuple[str, ...] = ("A", "B", "C")) -> dict[str, Any]:
        """Score A/B/C at the next token under the exact local-HF chat path.

        This diagnostic API shares checkpoint loading, image preparation, chat
        template use, processor invocation, and device placement with ``infer``.
        It does not alter ``infer`` or the semantic verifier protocol.
        """
        self.calls += 1
        contract = self.forced_choice_token_contract(request, choices=choices)
        if choice_token_ids is not None and choice_token_ids != contract["choice_token_ids"]:
            raise BackendError("CHOICE_TOKEN_CONTRACT_DRIFT")
        paths = request.get("image_paths", [])
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        try:
            with self.torch.inference_mode():
                output = self.model(**inputs)
                logits = output.logits[:, -1, :]
                log_probs = self.torch.log_softmax(logits, dim=-1)[0]
            values = {label: {"token_id": token_id,
                              "raw_logit": float(logits[0, token_id].detach().to("cpu").item()),
                              "log_probability": float(log_probs[token_id].detach().to("cpu").item())}
                      for label, token_id in contract["choice_token_ids"].items()}
        except Exception:
            raise BackendError("LOCAL_HF_CHOICE_SCORING_FAILURE") from None
        return {"choice_contract": contract, "choices": values}
