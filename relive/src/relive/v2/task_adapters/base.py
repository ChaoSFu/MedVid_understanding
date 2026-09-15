"""Task-adapter protocol: public question plus frozen ontology only."""
from __future__ import annotations
from typing import Protocol
from ..task_selection import PublicTaskQuestion
from ..temporal_localization import EventOntology, RequirementBuildResult

class TaskAdapter(Protocol):
    adapter_version: str
    def build_requirement(self, public_question: PublicTaskQuestion, ontology: EventOntology) -> RequirementBuildResult: ...
