from __future__ import annotations

import asyncio
import ast
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from ..config import REWARD_LMM, REWARD_LMM_PATH


def load_official_reward_prompt(dig_repo_dir: Path) -> str:
    utils_path = Path(dig_repo_dir).resolve() / "utils.py"
    tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "REWARD_ASSIGNMENT_PROMPT" in names:
                return str(ast.literal_eval(node.value))
    raise KeyError(f"REWARD_ASSIGNMENT_PROMPT not found in {utils_path}")


def encode_frame_base64(path: str) -> str:
    from PIL import Image

    buffer = BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_reward(text: str) -> float | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            value = json.loads(repair_json(text))
        except Exception:
            return None
    if isinstance(value, dict) and "reward" in value:
        return float(value["reward"])
    return None


class OfficialRewardAssigner:
    """Thin wrapper around DIG reward-assignment prompt and response semantics."""

    def __init__(
        self,
        dig_repo_dir: Path,
        model: str = REWARD_LMM_PATH,
        display_name: str = REWARD_LMM,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        concurrency: int = 64,
        max_retries: int = 5,
    ) -> None:
        self.dig_repo_dir = Path(dig_repo_dir)
        self.model = model
        self.display_name = display_name
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.prompt_template = load_official_reward_prompt(self.dig_repo_dir)

    async def score_one(
        self,
        client: Any,
        semaphore: asyncio.Semaphore,
        question: str,
        frame_path: str,
        duration: float,
        timestamp: float,
    ) -> float:
        async with semaphore:
            frame_b64 = await asyncio.to_thread(encode_frame_base64, frame_path)
            prompt = (
                self.prompt_template.replace("<<<question>>>", question)
                .replace("<<<duration>>>", str(round(float(duration), 2)))
                .replace("<<<timestamp>>>", str(round(float(timestamp), 2)))
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for _ in range(self.max_retries):
                try:
                    response = await client.chat.completions.create(model=self.model, messages=messages, temperature=0.7)
                    reward = parse_reward(response.choices[0].message.content or "")
                    if reward is not None:
                        return float(reward)
                except Exception:
                    await asyncio.sleep(1)
            return 0.0

    async def score_many(self, row: dict[str, Any], r_frame_positions: list[int]) -> list[float]:
        from openai import AsyncOpenAI

        observations = {int(obs["frame_position"]): obs for obs in row.get("frame_observations", [])}
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        semaphore = asyncio.Semaphore(max(1, self.concurrency))
        tasks = []
        for pos in r_frame_positions:
            obs = observations[int(pos)]
            tasks.append(
                self.score_one(
                    client=client,
                    semaphore=semaphore,
                    question=str(row.get("question", "")),
                    frame_path=str(obs["frame_path"]),
                    duration=float(row.get("clip_duration") or 0.0),
                    timestamp=float(obs["local_time"]),
                )
            )
        return [float(x) for x in await asyncio.gather(*tasks)]
