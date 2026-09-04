from __future__ import annotations

import asyncio
import ast
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..config import QUERY_IDENTIFIER_MODEL


def load_official_prompt(dig_repo_dir: Path) -> str:
    utils_path = Path(dig_repo_dir).resolve() / "utils.py"
    tree = ast.parse(utils_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "QUERY_IDENTIFICATION_PROMPT" in names:
                return str(ast.literal_eval(node.value))
    raise KeyError(f"QUERY_IDENTIFICATION_PROMPT not found in {utils_path}")


def parse_json_response(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json

            value = json.loads(repair_json(text))
        except Exception:
            return None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "isGlobal" in item:
                return item
        return None
    return value if isinstance(value, dict) else None


def parse_query_type(text: str) -> str | None:
    result = parse_json_response(text)
    if not result or "isGlobal" not in result:
        return None
    value = result["isGlobal"]
    if isinstance(value, str):
        if value.lower() == "true":
            return "global"
        if value.lower() == "false":
            return "local"
    if isinstance(value, bool):
        return "global" if value else "local"
    return None


class QueryIdentifierUnavailable(RuntimeError):
    pass


class OfficialQueryIdentifier:
    """Thin wrapper around DIG's official query-identification prompt/parser."""

    def __init__(
        self,
        dig_repo_dir: Path,
        model: str = QUERY_IDENTIFIER_MODEL,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "token-abc123",
        concurrency: int = 20,
        max_retries: int = 5,
    ) -> None:
        self.dig_repo_dir = Path(dig_repo_dir)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.prompt_template = load_official_prompt(self.dig_repo_dir)

    def preflight(self) -> dict[str, Any]:
        configured_model = self.model or os.environ.get("MODEL_NAME", "")
        server_status = self._server_models_status()
        ok = configured_model == QUERY_IDENTIFIER_MODEL and server_status["status"] == "OK"
        return {
            "status": "OK" if ok else "QUERY_IDENTIFIER_UNAVAILABLE",
            "required_model": QUERY_IDENTIFIER_MODEL,
            "configured_model": configured_model,
            "base_url": self.base_url,
            "server_models": server_status,
            "official_prompt_loaded": "<<<question>>>" in self.prompt_template,
            "replacement_policy": "No heuristic/model substitution is allowed without review.",
        }

    def _server_models_status(self) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return {"status": "UNAVAILABLE", "url": url, "error": repr(exc), "available_models": []}
        models = []
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item["id"]))
        has_required = self.model in models if models else True
        return {
            "status": "OK" if has_required else "MODEL_NOT_SERVED",
            "url": url,
            "available_models": models,
        }

    async def classify_one(self, client: Any, semaphore: asyncio.Semaphore, question: str) -> str:
        async with semaphore:
            prompt = self.prompt_template.replace("<<<question>>>", question)
            messages = [{"role": "user", "content": prompt}]
            last_error: BaseException | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                    )
                    text = response.choices[0].message.content or ""
                    query_type = parse_query_type(text)
                    if query_type in {"global", "local"}:
                        return query_type
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(1)
            raise QueryIdentifierUnavailable(f"Could not classify query after {self.max_retries} attempts: {last_error!r}")

    async def classify_many(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        preflight = self.preflight()
        if preflight["status"] != "OK":
            raise QueryIdentifierUnavailable(json.dumps(preflight, ensure_ascii=False))
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
        semaphore = asyncio.Semaphore(max(1, self.concurrency))
        tasks = [self.classify_one(client, semaphore, str(row.get("question", ""))) for row in rows]
        out = []
        for row, task in zip(rows, tasks):
            query_type = await task
            out.append(
                {
                    "sample_id": row["sample_id"],
                    "qa_type": row["qa_type"],
                    "dataset_name": row.get("dataset_name"),
                    "query_type": query_type,
                    "query_identifier_model": self.model,
                }
            )
        return out
