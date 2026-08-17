"""
Coding and Repository Tools for Galactus/Nanochat Agentic Workflows.

Provides:
- read_file: View file contents with line numbering and line range slicing
- write_file: Create or overwrite files with automatic directory scaffolding
- edit_file: Surgical search-and-replace text modifications
- grep_search: Search codebase for regex or string patterns
- list_files: Explore directory and file trees
"""

import os
import re
import fnmatch
from typing import Dict, Any, Optional, List


def _resolve_path(path: str) -> str:
    """Resolve and normalize a relative or absolute path."""
    if not path:
        return os.getcwd()
    return os.path.abspath(os.path.expanduser(path))


def execute_read_file(args: Dict[str, Any]) -> str:
    """
    Read file contents with line numbering and optional line range slicing.
    Usage: {"path": "file.py", "start_line": 1, "end_line": 50}
    """
    raw_path = args.get("path") or args.get("file") or args.get("filename")
    if not raw_path:
        return "[error] missing required argument 'path'"

    filepath = _resolve_path(raw_path)
    if not os.path.exists(filepath):
        return f"[error] file not found: {raw_path}"

    if not os.path.isfile(filepath):
        return f"[error] path is not a file: {raw_path}"

    start_line = int(args.get("start_line", 1))
    end_line = int(args.get("end_line", start_line + 299)) # default max 300 lines

    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"[error] failed to read file: {e}"

    total_lines = len(lines)
    if total_lines == 0:
        return f"(file {raw_path} is empty)"

    sliced_lines = lines[start_line - 1 : end_line]
    output_lines = []
    for idx, line in enumerate(sliced_lines, start=start_line):
        cleaned_line = line.rstrip('\r\n')
        output_lines.append(f"{idx}: {cleaned_line}")

    header = f"--- {raw_path} (lines {start_line}-{min(end_line, total_lines)} of {total_lines}) ---"
    return header + "\n" + "\n".join(output_lines)


def execute_write_file(args: Dict[str, Any]) -> str:
    """
    Create or overwrite a file.
    Usage: {"path": "file.py", "content": "print('hello')", "overwrite": true}
    """
    raw_path = args.get("path") or args.get("file") or args.get("filename")
    if not raw_path:
        return "[error] missing required argument 'path'"

    content = args.get("content", "")
    overwrite = bool(args.get("overwrite", True))

    filepath = _resolve_path(raw_path)

    if os.path.exists(filepath) and not overwrite:
        return f"[error] file already exists: {raw_path}. Set overwrite=true to replace."

    try:
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        line_count = len(content.splitlines())
        return f"Successfully wrote {len(content)} bytes ({line_count} lines) to {raw_path}"
    except Exception as e:
        return f"[error] failed to write file: {e}"


def execute_edit_file(args: Dict[str, Any]) -> str:
    """
    Surgical search-and-replace edit in a file.
    Usage: {"path": "file.py", "target_content": "old code", "replacement_content": "new code"}
    """
    raw_path = args.get("path") or args.get("file") or args.get("filename")
    if not raw_path:
        return "[error] missing required argument 'path'"

    target = args.get("target_content") or args.get("old_str") or args.get("target")
    if target is None:
        return "[error] missing required argument 'target_content'"

    replacement = args.get("replacement_content") or args.get("new_str") or args.get("replacement", "")
    allow_multiple = bool(args.get("allow_multiple", False))

    filepath = _resolve_path(raw_path)
    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        return f"[error] file not found: {raw_path}"

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except Exception as e:
        return f"[error] failed to read file: {e}"

    count = original.count(target)
    if count == 0:
        return f"[error] target_content not found in {raw_path}"

    if count > 1 and not allow_multiple:
        return (
            f"[error] target_content found {count} times in {raw_path}. "
            "Include more surrounding context to match uniquely, or set allow_multiple=true."
        )

    modified = original.replace(target, replacement)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(modified)
        return f"Successfully replaced {count} occurrence(s) in {raw_path}"
    except Exception as e:
        return f"[error] failed to save edited file: {e}"


def execute_grep_search(args: Dict[str, Any]) -> str:
    """
    Search for a text pattern or regex across files in a directory.
    Usage: {"query": "def train", "path": ".", "include": "*.py"}
    """
    query = args.get("query") or args.get("pattern")
    if not query:
        return "[error] missing required argument 'query'"

    search_path = _resolve_path(args.get("path", "."))
    include_filter = args.get("include") or args.get("glob")
    max_matches = int(args.get("max_matches", 30))

    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error:
        # Fall back to literal string search if not valid regex
        regex = re.compile(re.escape(query), re.IGNORECASE)

    matches = []
    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", ".cache"}

    if os.path.isfile(search_path):
        files_to_search = [search_path]
    else:
        files_to_search = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for file in files:
                if include_filter and not fnmatch.fnmatch(file, include_filter):
                    continue
                files_to_search.append(os.path.join(root, file))

    for filepath in files_to_search:
        if len(matches) >= max_matches:
            break
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if regex.search(line):
                        rel_path = os.path.relpath(filepath, os.getcwd())
                        matches.append(f"{rel_path}:{line_no}: {line.strip()}")
                        if len(matches) >= max_matches:
                            break
        except Exception:
            continue

    if not matches:
        return f"No matches found for '{query}'"

    res = "\n".join(matches)
    if len(matches) >= max_matches:
        res += f"\n... (reached max limit of {max_matches} matches)"
    return res


def execute_list_files(args: Dict[str, Any]) -> str:
    """
    List files and directories in a given path.
    Usage: {"path": ".", "pattern": "*.py"}
    """
    target_path = _resolve_path(args.get("path", "."))
    pattern = args.get("pattern") or args.get("glob")
    max_entries = int(args.get("max_entries", 50))

    if not os.path.exists(target_path):
        return f"[error] path not found: {args.get('path', '.')}"

    if os.path.isfile(target_path):
        return os.path.basename(target_path)

    ignore_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache", ".cache"}
    entries = []

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for name in sorted(dirs + files):
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, target_path)
            is_dir = os.path.isdir(full)
            entries.append(f"{rel}{'/' if is_dir else ''}")
            if len(entries) >= max_entries:
                break
        if len(entries) >= max_entries:
            break

    if not entries:
        return f"No files found in {args.get('path', '.')}"

    res = "\n".join(entries)
    if len(entries) >= max_entries:
        res += f"\n... (reached max limit of {max_entries} items)"
    return res
