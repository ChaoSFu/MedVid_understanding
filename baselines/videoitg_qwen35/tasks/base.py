from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import MAX_NEW_TOKENS_BY_TASK, TASK_NAME_BY_QA_TYPE


@dataclass(frozen=True)
class TaskInput:
    prompt: str
    max_new_tokens: int
    parser_status: str = "NOT_PARSED"
    parsed_prediction: Any | None = None


class TaskAdapter:
    task_name = "base"
    qa_types: set[str] = set()
    output_instruction = ""

    def applies_to(self, qa_type: str) -> bool:
        return qa_type in self.qa_types

    def build_prompt(self, manifest_row: dict[str, Any], frame_lines: list[str] | None = None) -> str:
        question = str(manifest_row.get("question", "") or "").replace("<video>\n", "").replace("<video>", "").strip()
        pieces: list[str] = []
        if frame_lines:
            pieces.append("Selected frames are shown in chronological order with clip-local timestamps:")
            pieces.extend(frame_lines)
            pieces.append("")
        pieces.append(question)
        if self.output_instruction:
            pieces.append("")
            pieces.append(self.output_instruction)
        return "\n".join(pieces).strip()

    def parse_prediction(self, text: str) -> tuple[str, Any | None]:
        if not text or not text.strip():
            return "EMPTY", None
        return "OK", text.strip()

    def max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS_BY_TASK[self.task_name]


class TALAdapter(TaskAdapter):
    task_name = "tal"
    qa_types = {"tal"}
    output_instruction = (
        "Return one or more temporal intervals in seconds as start-end seconds. "
        "Do not merge separate intervals unless the video evidence supports one continuous interval."
    )


class STGAdapter(TaskAdapter):
    task_name = "stg"
    qa_types = {"stg"}
    output_instruction = "Return the answer in the MedVidU STG format with time in seconds and bbox coordinates."


class DenseCaptionAdapter(TaskAdapter):
    task_name = "dvc"
    qa_types = {"dense_captioning_gpt", "dense_captioning_gemini"}
    output_instruction = "Return dense temporal captions using start-end seconds followed by a factual caption."


class NextActionAdapter(TaskAdapter):
    task_name = "next_action"
    qa_types = {"next_action"}
    output_instruction = "Return only the predicted next action."


class CVSAdapter(TaskAdapter):
    task_name = "cvs_assessment"
    qa_types = {"cvs_assessment"}
    output_instruction = (
        "Return CVS labels in the requested format, including Two structures, "
        "Cystic plate, and Hepatocystic triangle."
    )


class VideoSummaryAdapter(TaskAdapter):
    task_name = "vs"
    qa_types = {"video_summary_gpt", "video_summary_gemini"}
    output_instruction = "Return a concise surgical video summary."


class RegionCaptionAdapter(TaskAdapter):
    task_name = "rc"
    qa_types = {"region_caption_gpt", "region_caption_gemini"}
    output_instruction = "Return a concise description of the specified region only."


class SkillAssessmentAdapter(TaskAdapter):
    task_name = "skill_assessment"
    qa_types = {"skill_assessment"}
    output_instruction = "Return the six skill ratings in the requested 1/5 format."


ADAPTERS: tuple[TaskAdapter, ...] = (
    TALAdapter(),
    STGAdapter(),
    DenseCaptionAdapter(),
    NextActionAdapter(),
    CVSAdapter(),
    VideoSummaryAdapter(),
    RegionCaptionAdapter(),
    SkillAssessmentAdapter(),
)


def get_adapter(qa_type: str) -> TaskAdapter:
    expected = TASK_NAME_BY_QA_TYPE.get(qa_type)
    for adapter in ADAPTERS:
        if adapter.applies_to(qa_type):
            if expected is not None and adapter.task_name != expected:
                raise RuntimeError(f"Adapter dispatch mismatch for {qa_type}: {adapter.task_name} != {expected}")
            return adapter
    raise KeyError(f"Unsupported qa_type: {qa_type}")
