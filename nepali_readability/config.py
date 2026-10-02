"""Dataclasses/loaders for YAML configs in configs/."""
from dataclasses import dataclass, field
import yaml


@dataclass
class MethodConfig:
    name: str = ""
    params: dict = field(default_factory=dict)
    tokenizer: str = "naive"
    syllable_counter: str = "naive"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
