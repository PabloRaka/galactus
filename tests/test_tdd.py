"""
Tests for Autonomous TDD Loop and Sandboxed Test Runner.

python -m pytest tests/test_tdd.py -v
"""

from nanochat.tdd import run_isolated_tests, extract_code_blocks, TDD_SYSTEM_PROMPT


def test_extract_code_blocks():
    text = "Here is the code:\n```python\ndef is_prime(n):\n    return n > 1\n```\nAnd tests:\n```python\nassert is_prime(2)\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert "def is_prime" in blocks[0]
    assert "assert is_prime(2)" in blocks[1]


def test_run_isolated_tests_passing():
    code = """
def add(a, b):
    return a + b
"""
    test_code = """
assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
"""
    res = run_isolated_tests(code, test_code)
    assert res["success"] is True
    assert "passed successfully" in res["summary"]
    assert res["error"] == ""


def test_run_isolated_tests_failing_assertion():
    code = """
def multiply(a, b):
    return a + b  # Intentional bug
"""
    test_code = """
assert multiply(2, 3) == 6
"""
    res = run_isolated_tests(code, test_code)
    assert res["success"] is False
    assert "AssertionError" in res["error"] or "AssertionError" in res["summary"]


def test_run_isolated_tests_syntax_error():
    code = """
def broken_fn(:
    pass
"""
    test_code = "assert True"
    res = run_isolated_tests(code, test_code)
    assert res["success"] is False
    assert "SyntaxError" in res["error"] or "failed" in res["summary"].lower()


def test_tdd_system_prompt_structure():
    assert "RED" in TDD_SYSTEM_PROMPT
    assert "GREEN" in TDD_SYSTEM_PROMPT
    assert "REFACTOR" in TDD_SYSTEM_PROMPT
    assert "assert" in TDD_SYSTEM_PROMPT
