"""Workspace path confinement for the code assistant.

Every file path the assistant reads or writes is resolved against
`settings.PROJECT_ROOT` and must stay inside it; a path that escapes the workspace
(e.g. `../secret.txt`) is rejected.
"""

from pathlib import Path

from core import settings


class PathEscapesWorkspace(RuntimeError):
    pass


def resolve_workspace_path(user_path: str) -> Path:
    root = Path(settings.PROJECT_ROOT).expanduser().resolve()
    candidate = (root / user_path).resolve()
    if not candidate.is_relative_to(root):
        raise PathEscapesWorkspace(f"path {user_path!r} escapes workspace root {root}")
    return candidate
