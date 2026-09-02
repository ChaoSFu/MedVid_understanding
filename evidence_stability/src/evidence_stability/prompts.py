from __future__ import annotations

import re


PROMPT_VERSION = "evidence_presence_v1"


def build_evidence_presence_prompt(target_action: str) -> str:
    return f"""You are examining a short temporal window from a medical or surgical video.

Target action:
"{target_action}"

Does this video window contain visual evidence that the target action is actually being performed?

Answer only:
YES
or
NO

Judge only what is visually observable in the provided frames.

Important:

* Do not infer the action from what may happen before or after this window.
* Do not answer YES merely because a relevant instrument or anatomical structure is present.
* Answer YES only when the target action itself is visually supported by the frames."""


def parse_yes_no(raw_response: str | None) -> str:
    text = (raw_response or "").strip()
    normalized = re.sub(r"\s+", " ", text).strip()
    if re.fullmatch(r"yes[.!]?", normalized, flags=re.IGNORECASE):
        return "YES"
    if re.fullmatch(r"no[.!]?", normalized, flags=re.IGNORECASE):
        return "NO"
    return "INVALID"
