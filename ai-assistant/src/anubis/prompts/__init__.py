"""Prompt registry — loads and renders prompt templates.

Templates are stored as TOML files with Jinja2 placeholders, keeping
all prompt text outside of Python business logic.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from jinja2 import BaseLoader, Environment, TemplateNotFound

_TEMPLATE_DIR = Path(__file__).parent


class PromptRegistry:
    """Manages prompt templates and handles Jinja2 rendering."""

    def __init__(self, template_path: Path | None = None) -> None:
        path = template_path or (_TEMPLATE_DIR / "templates.toml")
        with path.open("rb") as f:
            self._templates: dict[str, dict[str, str]] = tomllib.load(f)
        self._env = Environment(loader=BaseLoader(), autoescape=False)  # noqa: S701

    def render_prompt(self, key: str, **variables: object) -> str:
        """Fetch and render a prompt template identified by its dotted key."""
        parts = key.split(".")
        node: dict[str, object] | str = self._templates
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]  # type: ignore[assignment]
            else:
                raise TemplateNotFound(key)

        if not isinstance(node, str):
            if isinstance(node, dict) and "template" in node:
                node = node["template"]  # type: ignore[assignment]
            else:
                raise TemplateNotFound(key)

        template = self._env.from_string(str(node))
        return template.render(**variables)

    @property
    def available_keys(self) -> list[str]:
        """Return a list of all available prompt template keys."""
        keys: list[str] = []
        self.gather_template_keys(self._templates, "", keys)
        return keys

    def gather_template_keys(
        self, node: dict[str, object], prefix: str, acc: list[str]
    ) -> None:
        """Recursively traverse the template structure to collect all valid keys."""
        for k, v in node.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                if "template" in v:
                    acc.append(full)
                else:
                    self.gather_template_keys(v, full, acc)
            elif isinstance(v, str):
                acc.append(full)


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """Singleton accessor for the default PromptRegistry."""
    return PromptRegistry()
