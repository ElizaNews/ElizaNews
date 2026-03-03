"""Skill registry — dispatch skills by name or alias."""

from __future__ import annotations

from training.rl.skills.base_skill import BaseSkill


# Alias map: common phrases → canonical skill name.
DEFAULT_ALIASES: dict[str, str] = {
    "go forward": "walk",
    "go backwards": "walk",
    "move forward": "walk",
    "move": "walk",
    "walk forward": "walk",
    "walk backward": "walk",
    "walk slowly": "walk",
    "walk fast": "walk",
    "stop": "stand",
    "halt": "stand",
    "freeze": "stand",
    "turn left": "turn",
    "turn right": "turn",
    "rotate": "turn",
    "spin": "turn",
    "wave hello": "wave",
    "wave hand": "wave",
    "say hello": "wave",
    "greet": "wave",
    "bow down": "bow",
    "take a bow": "bow",
}


class SkillRegistry:
    """Registry of available skills with alias support."""

    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}
        self._aliases: dict[str, str] = dict(DEFAULT_ALIASES)

    def register(self, skill: BaseSkill) -> None:
        """Register a skill by its name."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill | None:
        """Look up a skill by name or alias."""
        # Direct lookup.
        if name in self._skills:
            return self._skills[name]
        # Alias lookup.
        canonical = self._aliases.get(name.lower())
        if canonical and canonical in self._skills:
            return self._skills[canonical]
        return None

    def list_skills(self) -> list[str]:
        """Return all registered skill names."""
        return sorted(self._skills.keys())

    def add_alias(self, alias: str, skill_name: str) -> None:
        """Add a custom alias mapping."""
        self._aliases[alias.lower()] = skill_name

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        return len(self._skills)
