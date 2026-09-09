"""Task extension points without imports of historical annotation loaders."""
from __future__ import annotations

from typing import Protocol

from relive.data.schemas import SUPPORTED_TASKS, load_runtime
from relive.types import RuntimeSample


class PublicRuntimeAdapter(Protocol):
    def load(self, public_runtime_path: str) -> list[RuntimeSample]: ...


def task_support(task: str) -> dict:
    return {"task": task, "status": "SUPPORTED" if task in SUPPORTED_TASKS else "UNSUPPORTED",
            "reason": None if task in SUPPORTED_TASKS else "ReliVE-v1 implements claim_verification and action_qa only."}


class STGAdapter:
    """A deliberately limited interface, not an official STG evaluator.

    Historical read-only schema audit identifies the public human question and
    ordered video frames; assistant conversation answers and struc_info must stay
    outside runtime. No official file is opened by this class.
    """
    schema_audit_reference = "outputs/stg_pilot/phase_g/h4_spatial_v1/audit/medvidu_stg_schema.md"

    def load(self, public_runtime_path: str) -> list[RuntimeSample]:
        samples = load_runtime(public_runtime_path)
        if any(s.task != "stg" for s in samples):
            raise ValueError("STGAdapter accepts only explicitly tagged stg samples")
        return samples

    def status(self) -> dict:
        return {"task": "stg", "status": "UNSUPPORTED", "schema_inspected": True,
                "schema_audit_reference": self.schema_audit_reference,
                "official_runtime_connected": False, "official_prediction_supported": False,
                "reason": "Public-only source conversion and task-specific official prediction are not connected; no action_qa substitution."}
