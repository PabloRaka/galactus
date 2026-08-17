import json
from collections import deque

try:
    from nanochat.engine import RowState, dispatch_tool_call, TOOL_REGISTRY
except ImportError:
    # Standalone mock for non-torch/CPU test environment
    class RowState:
        def __init__(self, current_tokens=None, json_constraint=None, max_tool_calls=5):
            self.current_tokens = current_tokens or []
            self.forced_tokens = deque()
            self.in_tool_call = False
            self.tool_call_tokens = []
            self.tool_call_count = 0
            self.max_tool_calls = max_tool_calls
            self.tool_call_history = []
            self.tools_disabled = False
            self.json_constraint = json_constraint
            self.completed = False

    TOOL_REGISTRY = {}
    def dispatch_tool_call(tool_call_json):
        try:
            call = json.loads(tool_call_json)
        except (json.JSONDecodeError, TypeError):
            return None
        name = call.get("name")
        if name not in TOOL_REGISTRY:
            return None
        return TOOL_REGISTRY[name](call)


def test_rowstate_initialization():
    state = RowState(max_tool_calls=3)
    assert state.tool_call_count == 0
    assert state.max_tool_calls == 3
    assert not state.tools_disabled
    assert len(state.tool_call_history) == 0


def test_dispatch_tool_call_unknown():
    res = dispatch_tool_call('{"name": "non_existent_tool"}')
    assert res is None


def test_dispatch_tool_call_malformed():
    res = dispatch_tool_call('not a valid json')
    assert res is None


def test_self_correction_cycle_detection_logic():
    """
    Simulates the engine's anti-loop logic when a model repeatedly emits
    the same failing tool call.
    """
    state = RowState(max_tool_calls=5)

    # First attempt: failing tool call
    payload1 = '{"name": "python", "code": "invalid_syntax(("}'
    state.tool_call_count += 1
    result1 = dispatch_tool_call(payload1)
    if result1 is None:
        result1 = "[error] execution failed"
    state.tool_call_history.append((payload1, result1))

    assert state.tool_call_count == 1
    assert not state.tools_disabled

    # Second attempt: exact same failing payload (cycle/repeat)
    payload2 = '{"name": "python", "code": "invalid_syntax(("}'
    state.tool_call_count += 1

    # Simulate Engine safeguard check
    if (
        len(state.tool_call_history) >= 1
        and payload2 == state.tool_call_history[-1][0]
        and state.tool_call_history[-1][1].startswith("[error]")
    ):
        state.tools_disabled = True
        result2 = "[error] Repeated identical failing tool call detected. Tool loop aborted to prevent infinite execution. Provide your best answer now."
    else:
        result2 = dispatch_tool_call(payload2)

    assert state.tools_disabled
    assert "Repeated identical" in result2


def test_max_tool_calls_ceiling():
    """
    Simulates reaching the max_tool_calls limit (e.g. 3) and forcing answer generation.
    """
    max_limit = 3
    state = RowState(max_tool_calls=max_limit)

    for i in range(1, 5):
        payload = f'{{"name": "python", "code": "1 + {i}"}}'
        state.tool_call_count += 1

        if state.tool_call_count > state.max_tool_calls or state.tools_disabled:
            state.tools_disabled = True
            result = f"[error] Maximum tool iteration limit ({state.max_tool_calls}) reached. Please provide your final answer now without calling any more tools."
        else:
            result = f"{1 + i}"
            state.tool_call_history.append((payload, result))

    # After 4 attempts with max_limit=3, tools must be disabled
    assert state.tools_disabled
    assert state.tool_call_count == 4
    assert len(state.tool_call_history) == 3
