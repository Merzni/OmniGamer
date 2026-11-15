"""Shared result and descriptor contracts for every explicit skill."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field


class SkillStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    ABORT = "ABORT"
    UNCONFIRMED = "UNCONFIRMED"


class SkillResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: SkillStatus
    result_code: str = ""
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0, ge=0)


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    params_model: type[BaseModel]
    preconditions: tuple[str, ...]
    result_contract: tuple[str, ...]
    timeout_seconds: float
    resources: tuple[str, ...]
    executor: Callable[..., SkillResult]

    def planner_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_model.model_json_schema(),
            "preconditions": list(self.preconditions),
            "results": list(self.result_contract),
            "timeout_seconds": self.timeout_seconds,
            "resources": list(self.resources),
        }

    def validate_args(self, args):
        return self.params_model.model_validate(args)


class SkillRegistry:
    def __init__(self, specs=()):
        self._specs: dict[str, SkillSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: SkillSpec):
        if spec.name in self._specs:
            raise ValueError(f"Skill already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ValueError(f"Unknown skill: {name}") from exc

    def validate(self, name: str, args):
        return self.get(name).validate_args(args)

    def planner_catalog(self):
        return [self._specs[name].planner_schema() for name in sorted(self._specs)]

    def names(self):
        return tuple(sorted(self._specs))
