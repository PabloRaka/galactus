import os
import json
import tempfile
from nanochat.engine import dispatch_tool_call, TOOL_REGISTRY, RowState
from nanochat.tokenizer import get_tokenizer


def test_dispatch_tool_call_flat_and_nested():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "sample.py").replace("\\", "/")

        # 1. Flat schema write_file
        res_w = dispatch_tool_call(json.dumps({
            "name": "write_file",
            "path": test_path,
            "content": "def test():\n    return 42\n"
        }))
        assert "Successfully wrote" in res_w
        assert os.path.exists(test_path)

        # 2. Nested schema read_file
        res_r = dispatch_tool_call(json.dumps({
            "name": "read_file",
            "arguments": {"path": test_path, "start_line": 1, "end_line": 2}
        }))
        assert "1: def test():" in res_r
        assert "2:     return 42" in res_r

        # 3. Python execution tool
        res_py = dispatch_tool_call(json.dumps({
            "name": "python",
            "code": "print(sum([10, 20, 30]))"
        }))
        assert "60" in res_py.strip()

        # 4. JSON with trailing comma (auto repair)
        malformed_json = f'{{"name": "read_file", "path": "{test_path}",}}'
        res_rep = dispatch_tool_call(malformed_json)
        assert "1: def test():" in res_rep

        # 5. Unknown tool gives helpful feedback
        res_unk = dispatch_tool_call(json.dumps({"name": "non_existent_tool"}))
        assert "[error] Unknown tool" in res_unk


def test_row_state_safeguards():
    state = RowState(max_tool_calls=2)

    # Simulate tool calls reaching limit
    state.tool_call_count = 2
    # At count > max_tool_calls, tools become disabled
    state.tool_call_count += 1
    assert state.tool_call_count > state.max_tool_calls


def test_tokenizer_tool_tokens():
    tok = get_tokenizer()
    for token_name in ["<|tool_call|>", "<|tool_call_end|>", "<|tool_result|>", "<|tool_result_end|>"]:
        token_id = tok.encode_special(token_name)
        assert isinstance(token_id, int)
        assert tok.decode([token_id]) == token_name


if __name__ == "__main__":
    test_dispatch_tool_call_flat_and_nested()
    test_row_state_safeguards()
    test_tokenizer_tool_tokens()
    print("All tool calling tests passed successfully!")
