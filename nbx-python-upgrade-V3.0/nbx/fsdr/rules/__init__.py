"""Rule registries (master §4): one YAML per rule set. V3.0 adds the four group rule sets of the update prompt."""
from __future__ import annotations

from pathlib import Path

import yaml

FOLDER = Path(__file__).parent
GROUP_REGISTRIES = {"t1c": "production_rules_constraint.yaml", "t1f": "production_rules_free.yaml",
                    "t2c": "challenger_rules_constraint_v1.yaml", "t2f": "challenger_rules_free_v1.yaml"}


def load(name: str) -> dict:
    return yaml.safe_load((FOLDER / name).read_text(encoding="utf-8"))


def group_registry(group: str) -> dict:
    return load(GROUP_REGISTRIES[group])
