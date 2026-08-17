"""
Autonomous Test-Driven Development (TDD) Orchestration for Galactus/Nanochat.

Provides:
- run_isolated_tests: Sandboxed test runner executing test suites against code implementations
- extract_code_blocks: Helper to parse python blocks and assert suites from markdown
- TDD_SYSTEM_PROMPT: Specialized system prompt enforcing Red-Green-Refactor protocol
"""

import re
from typing import Dict, Any, List, Optional
from nanochat.execution import execute_code


TDD_SYSTEM_PROMPT = """You are an elite software engineering assistant practicing strict Autonomous Test-Driven Development (TDD).

For every coding request, you must follow the Red-Green-Refactor cycle:

1. 🔴 RED (Test First):
   - In <|thought|>, analyze the specification and brainstorm normal, boundary, and edge cases.
   - Write comprehensive unit tests with `assert` statements before or alongside the implementation.
   - Use the `python` or `bash` tool to execute the test suite.

2. 🟢 GREEN (Make It Pass):
   - Write the minimal, robust implementation to satisfy the test cases.
   - Run the test suite via the `python` tool to verify correctness.

3. 🔄 REFACTOR & SELF-CORRECTION:
   - If any test fails (AssertionError, edge case bug, timeout), analyze the exact failure inside <|thought|>.
   - Use `edit_file` or run corrected code until all assertions pass cleanly.

4. ✨ VERIFIED DELIVERY:
   - Only deliver your final solution after all tests have passed. Include the verified test suite in your response.

You have full access to tools: python, bash, web_search, read_file, write_file, edit_file, grep_search, list_files."""


def extract_code_blocks(text: str, language: str = "python") -> List[str]:
    """
    Extract code blocks wrapped in markdown fences (```python ... ```).
    If no fences are found, returns the stripped text as a single candidate block.
    """
    if not text:
        return []

    pattern = rf'```(?:{language})?\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    stripped = text.strip()
    return [stripped] if stripped else []


def run_isolated_tests(
    implementation_code: str,
    test_code: str,
    timeout: int = 5,
) -> Dict[str, Any]:
    """
    Execute a test suite against an implementation in an isolated sandbox.

    Args:
        implementation_code: The source code implementing the requested logic.
        test_code: The unit tests (assert statements, pytest, or unittest).
        timeout: Execution timeout in seconds.

    Returns:
        Dict containing:
        - success: bool (True if all tests ran with exit code 0 and no exceptions)
        - output: str (Captured stdout)
        - error: str (Captured stderr or traceback)
        - summary: str (Short human-readable result)
    """
    full_program = (
        implementation_code.strip()
        + "\n\n# --- Unit Tests ---\n"
        + test_code.strip()
    )

    result = execute_code(full_program, timeout=timeout)

    stdout = result.stdout.strip() if result.stdout else ""
    stderr = result.stderr.strip() if result.stderr else ""
    error = (result.error.strip() if result.error else "") or stderr

    if result.success:
        summary = "✓ All unit test assertions passed successfully."
    elif result.timeout:
        summary = f"✗ Test execution timed out ({timeout}s limit)."
    elif "AssertionError" in error or "AssertionError" in stderr:
        summary = f"✗ Test failed: AssertionError encountered during validation.\n{error}"
    else:
        summary = f"✗ Execution failed: {error if error else 'Unknown runtime error'}"

    return {
        "success": result.success,
        "stdout": stdout,
        "stderr": stderr,
        "error": error,
        "summary": summary,
        "full_program": full_program,
    }
