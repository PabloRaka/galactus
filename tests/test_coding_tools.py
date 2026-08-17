"""
Tests for Coding Tools (read_file, write_file, edit_file, grep_search, list_files).

python -m pytest tests/test_coding_tools.py -v
"""

import os
import tempfile
from nanochat.coding_tools import (
    execute_read_file,
    execute_write_file,
    execute_edit_file,
    execute_grep_search,
    execute_list_files,
)


def test_write_and_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "subdir", "test.py")
        content = "def hello():\n    print('world')\n    return 42\n"

        # Test write_file
        res_write = execute_write_file({"path": test_file, "content": content, "overwrite": True})
        assert "Successfully wrote" in res_write
        assert os.path.exists(test_file)

        # Test read_file full
        res_read = execute_read_file({"path": test_file, "start_line": 1, "end_line": 3})
        assert "1: def hello():" in res_read
        assert "2:     print('world')" in res_read
        assert "3:     return 42" in res_read

        # Test read_file slicing
        res_slice = execute_read_file({"path": test_file, "start_line": 2, "end_line": 2})
        assert "2:     print('world')" in res_slice
        assert "1: def hello():" not in res_slice


def test_write_file_overwrite_guard():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "locked.txt")
        execute_write_file({"path": test_file, "content": "original", "overwrite": True})

        # Try to write without overwrite
        res_blocked = execute_write_file({"path": test_file, "content": "new", "overwrite": False})
        assert "[error] file already exists" in res_blocked


def test_edit_file_surgical():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "calculator.py")
        content = "def add(a, b):\n    return a - b  # bug here\n"
        execute_write_file({"path": test_file, "content": content})

        # Fix bug using edit_file
        res_edit = execute_edit_file({
            "path": test_file,
            "target_content": "return a - b  # bug here",
            "replacement_content": "return a + b",
        })
        assert "Successfully replaced 1 occurrence" in res_edit

        # Read back to verify
        read_back = execute_read_file({"path": test_file})
        assert "return a + b" in read_back
        assert "return a - b" not in read_back


def test_edit_file_target_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.txt")
        execute_write_file({"path": test_file, "content": "hello world"})

        res = execute_edit_file({
            "path": test_file,
            "target_content": "goodbye universe",
            "replacement_content": "something else",
        })
        assert "[error] target_content not found" in res


def test_grep_search_and_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "module_a.py")
        f2 = os.path.join(tmpdir, "module_b.py")
        execute_write_file({"path": f1, "content": "def calculate_loss():\n    pass\n"})
        execute_write_file({"path": f2, "content": "def calculate_reward():\n    pass\n"})

        # Test grep_search
        grep_res = execute_grep_search({"query": "calculate_loss", "path": tmpdir})
        assert "module_a.py:1: def calculate_loss():" in grep_res
        assert "module_b.py" not in grep_res

        # Test list_files
        list_res = execute_list_files({"path": tmpdir, "pattern": "*.py"})
        assert "module_a.py" in list_res
        assert "module_b.py" in list_res
