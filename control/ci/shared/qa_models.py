from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

class SuiteConfig(BaseModel):
    name: str = ""
    description: str = ""
    type: Literal["test", "lint"] = "test"
    environment: str | None = None
    requires_docker: bool = False
    compose_file: str | None = None
    profiles: list[str] = Field(default_factory=list)
    service: str | None = None
    test_dir: str | None = None
    parallel: bool = False
    pytest_args: list[str] = Field(default_factory=list)
    pre_run: str | None = None
    tasks: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

class EnvironmentConfig(BaseModel):
    """Maps to [environments.X] in qa.toml."""
    config_dir: str # Path to PANOSETI JSON configs for this env
    compose_file: str # Compose file for this environment

class QAConfig(BaseModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    environments: dict[str, EnvironmentConfig] = Field(default_factory=dict)
    suites: dict[str, SuiteConfig]

    @field_validator("suites", mode="before")
    @classmethod
    def inject_names(cls, v: Any) -> Any:
        if isinstance(v, dict):
            for name, config in v.items():
                if isinstance(config, dict) and "name" not in config:
                    config["name"] = name
        return v
