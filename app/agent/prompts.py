"""Load versioned research system prompts from prompts/ (Phase 11)."""

from __future__ import annotations

from pathlib import Path

from app.config import settings


def prompts_root(prompts_dir: str | Path | None = None) -> Path:
    raw = Path(prompts_dir) if prompts_dir is not None else Path(settings.PROMPTS_DIR)
    return raw if raw.is_absolute() else (Path.cwd() / raw).resolve()


def load_research_prompt(
    version: str | None = None,
    *,
    prompts_dir: str | Path | None = None,
) -> str:
    """Return the body of ``prompts/{version}.md``.

    ``version`` defaults to ``settings.PROMPT_VERSION`` (e.g. ``prompt_v1``).
    Raises ``FileNotFoundError`` with a clear message when the file is missing.
    """
    name = (version if version is not None else settings.PROMPT_VERSION).strip()
    if not name:
        raise ValueError("PROMPT_VERSION / version must be a non-empty string")
    # Allow bare names like prompt_v1; reject path traversal.
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"Invalid prompt version name: {name!r}")

    path = prompts_root(prompts_dir) / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(
            f"Research prompt not found: {path} "
            f"(set PROMPT_VERSION to an existing file under {prompts_root(prompts_dir)})"
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Research prompt file is empty: {path}")
    return text


def research_prompt_path(version: str | None = None) -> Path:
    name = (version if version is not None else settings.PROMPT_VERSION).strip()
    return prompts_root() / f"{name}.md"
