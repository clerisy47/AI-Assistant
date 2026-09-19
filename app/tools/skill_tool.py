"""Progressive-disclosure skill loading for the verified-research path.

System prompts list skill titles + one-line descriptions only. The full
SKILL.md body enters context after the model calls `load_skill(name)`.
Not registered on `/chat` — Phase 3 attaches this via `register_skill_tools`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.llm.base import ToolDefinition
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_SKILL_FILENAME = "SKILL.md"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    body: str
    path: Path


def _skills_root(skills_dir: str | Path | None = None) -> Path:
    raw = Path(skills_dir) if skills_dir is not None else Path(settings.SKILLS_DIR)
    return raw if raw.is_absolute() else (Path.cwd() / raw).resolve()


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a simple YAML frontmatter block without PyYAML.

    Supports scalar values and folded (`>-` / `>`) or literal (`|`) blocks
    for `description`. Returns (fields, body_markdown).
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    fm_raw = parts[1].strip("\n")
    body = parts[2].lstrip("\n")
    fields: dict[str, str] = {}
    lines = fm_raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.strip()
        if rest in (">", ">-", "|", "|-"):
            block: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t")):
                block.append(lines[i].strip())
                i += 1
            fields[key] = " ".join(block).strip()
            continue
        fields[key] = rest.strip().strip("\"'")
        i += 1
    return fields, body


def _load_skill_file(path: Path) -> SkillMeta | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read skill file %s: %s", path, exc)
        return None

    fields, body = _parse_frontmatter(text)
    name = (fields.get("name") or path.parent.name).strip()
    description = (fields.get("description") or "").strip()
    if not name or not _NAME_RE.match(name):
        logger.warning("Skipping skill with invalid name %r at %s", name, path)
        return None
    if not description:
        description = f"Skill '{name}' (no description in frontmatter)."
    return SkillMeta(name=name, description=description, body=body.strip(), path=path)


def discover_skills(skills_dir: str | Path | None = None) -> list[SkillMeta]:
    """Scan `*/SKILL.md` under the skills root; return sorted by name."""
    root = _skills_root(skills_dir)
    if not root.is_dir():
        return []

    found: list[SkillMeta] = []
    for skill_path in sorted(root.glob(f"*/{_SKILL_FILENAME}")):
        meta = _load_skill_file(skill_path)
        if meta is not None:
            found.append(meta)
    return found


def get_skill(name: str, skills_dir: str | Path | None = None) -> SkillMeta | None:
    """Return a skill by exact name, or None if missing / invalid name."""
    if not name or not _NAME_RE.match(name):
        return None
    for skill in discover_skills(skills_dir):
        if skill.name == name:
            return skill
    return None


def skill_catalog_section(skills_dir: str | Path | None = None) -> str:
    """Markdown list of skill name + one-line description only (no full body)."""
    skills = discover_skills(skills_dir)
    if not skills:
        return "(No skills available.)"
    lines = []
    for skill in skills:
        lines.append(f"- **{skill.name}**: {skill.description}")
    return "\n".join(lines)


def build_skills_system_prefix(skills_dir: str | Path | None = None) -> str:
    """Catalog-only system prefix for the research agent (Phase 3 will prepend).

    Full procedure bodies are intentionally omitted; the model must call
    `load_skill` before following a skill's steps.
    """
    catalog = skill_catalog_section(skills_dir)
    return (
        "Available skills (progressive disclosure):\n"
        f"{catalog}\n\n"
        "Call `load_skill` with a skill name before following that skill's "
        "full procedure. Do not invent procedure steps from the catalog alone."
    )


def _validate_skill_name(name: str) -> str | None:
    """Return an error message if `name` is unsafe or malformed; else None."""
    if not name or not isinstance(name, str):
        return "Error: skill name is required."
    cleaned = name.strip()
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        return f"Error: invalid skill name '{name}' (path separators not allowed)."
    if not _NAME_RE.match(cleaned):
        return (
            f"Error: invalid skill name '{name}'. "
            "Use a lowercase identifier like 'verified_research'."
        )
    return None


async def load_skill(name: str) -> str:
    """Return the full SKILL.md body for `name`, or an Error: string."""
    err = _validate_skill_name(name)
    if err:
        raise ValueError(err)

    skill = get_skill(name.strip())
    if skill is None:
        available = ", ".join(s.name for s in discover_skills()) or "(none)"
        raise ValueError(
            f"Error: unknown skill '{name}'. Available skills: {available}."
        )
    return skill.body


LOAD_SKILL_DEFINITION = ToolDefinition(
    name="load_skill",
    description=(
        "Load the full body of a named progressive-disclosure skill "
        "(procedure steps, anti-patterns). Call this when the catalog "
        "one-liner is not enough and you need the complete skill instructions. "
        "Pass the skill name exactly as listed in Available skills."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill identifier, e.g. 'verified_research'.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
)


def register_skill_tools(registry: ToolRegistry) -> None:
    """Register `load_skill` on a research-path ToolRegistry (not `/chat`)."""
    registry.register(LOAD_SKILL_DEFINITION, load_skill)
