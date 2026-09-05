"""Built-in code tools for the code assistant.

Every tool confines its file paths to the workspace rooted at `settings.PROJECT_ROOT`;
a path that escapes that root is rejected. The executor applies the same confinement
as defense-in-depth, so these tools also accept already-resolved absolute paths.
"""

import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from agents.code_assistant.sandbox import PathEscapesWorkspace
from core import settings


def _root() -> Path:
    return Path(settings.PROJECT_ROOT).expanduser().resolve()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _root() / p
    p = p.resolve()
    if not p.is_relative_to(_root()):
        raise PathEscapesWorkspace(f"path {path!r} escapes workspace root {_root()}")
    return p


def _read_file_func(path: str) -> str:
    """Read a file from the workspace and return its contents."""
    return _resolve(path).read_text(encoding="utf-8", errors="replace")


def _write_file_func(path: str, content: str) -> str:
    """Create or overwrite a file in the workspace with the given content."""
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {target}"


def _edit_file_func(path: str, old_string: str, new_string: str) -> str:
    """Replace the first exact occurrence of old_string with new_string in a file."""
    target = _resolve(path)
    text = target.read_text(encoding="utf-8", errors="replace")
    if old_string not in text:
        return f"Error: old_string not found in {target}"
    target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Edited {target}"


def _list_directory_func(path: str = ".") -> str:
    """List the entries directly under a workspace path."""
    target = _resolve(path)
    if not target.is_dir():
        return f"Error: {target} is not a directory"
    lines = []
    for entry in sorted(target.iterdir(), key=lambda e: e.name.lower()):
        lines.append(f"{'[DIR] ' if entry.is_dir() else '[FILE]'} {entry.name}")
    return "\n".join(lines) or "(empty)"


def _search_files_func(query: str, path: str = ".") -> str:
    """Recursively search file contents under a workspace path for a literal string.

    Returns matching `file:line` entries; skips binary and oversized files.
    """
    root = _resolve(path)
    matches: list[str] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            if candidate.stat().st_size > 1_000_000:
                continue
            for lineno, line in enumerate(
                candidate.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if query in line:
                    matches.append(f"{candidate.relative_to(root)}:{lineno}: {line.strip()}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(matches) or "(no matches)"


def _run_command_func(command: str) -> str:
    """Run a shell command in the workspace directory and return its combined output."""
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(_root()), capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out"
    parts = [p.strip() for p in (proc.stdout, proc.stderr) if p and p.strip()]
    return "\n".join(parts) or "(no output)"


def _move_file_func(source: str, destination: str) -> str:
    """Move or rename a file within the workspace."""
    src, dst = _resolve(source), _resolve(destination)
    if not src.exists():
        return f"Error: source {src} does not exist"
    src.rename(dst)
    return f"Moved {src} -> {dst}"


def _copy_file_func(source: str, destination: str) -> str:
    """Copy a file within the workspace."""
    src, dst = _resolve(source), _resolve(destination)
    shutil.copy2(src, dst)
    return f"Copied {src} -> {dst}"


def _delete_file_func(path: str) -> str:
    """Delete a file (or an empty directory) from the workspace."""
    target = _resolve(path)
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    return f"Deleted {target}"


read_file: BaseTool = tool(_read_file_func)
read_file.name = "ReadFile"

write_file: BaseTool = tool(_write_file_func)
write_file.name = "WriteFile"

edit_file: BaseTool = tool(_edit_file_func)
edit_file.name = "EditFile"

list_directory: BaseTool = tool(_list_directory_func)
list_directory.name = "ListDirectory"

search_files: BaseTool = tool(_search_files_func)
search_files.name = "SearchFiles"

run_command: BaseTool = tool(_run_command_func)
run_command.name = "RunCommand"

move_file: BaseTool = tool(_move_file_func)
move_file.name = "MoveFile"

copy_file: BaseTool = tool(_copy_file_func)
copy_file.name = "CopyFile"

delete_file: BaseTool = tool(_delete_file_func)
delete_file.name = "DeleteFile"


CODE_TOOLS: list[BaseTool] = [
    read_file,
    write_file,
    edit_file,
    list_directory,
    search_files,
    run_command,
    move_file,
    copy_file,
    delete_file,
]
