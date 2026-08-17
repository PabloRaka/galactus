"""
Test the tokenizer: BPE round-trips, special tokens, and conversation rendering.
Trains a tiny throwaway tokenizer in-process so the test is hermetic
(no dependency on ~/.cache/nanochat).

python -m pytest tests/test_tokenizer.py -v
"""

import pytest
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

# a small corpus is enough to exercise the BPE machinery
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "hello world, hello tokenizer, hello hello hello",
    "Numbers like 12345 and unicode like naïve café 你好 🙂 should survive.",
    "def f(x):\n    return x + 1\n",
] * 8


@pytest.fixture(scope="module")
def tokenizer():
    vocab_size = 256 + len(SPECIAL_TOKENS) + 35 # bytes + specials + a few merges
    return RustBPETokenizer.train_from_iterator(iter(CORPUS), vocab_size)


def test_vocab_size(tokenizer):
    assert tokenizer.get_vocab_size() == 256 + len(SPECIAL_TOKENS) + 35


def test_encode_decode_roundtrip(tokenizer):
    for text in ["hello world", "naïve café 你好 🙂", "unseen tokens: zqxjkv"]:
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text


def test_special_tokens(tokenizer):
    # all special tokens encode to a unique single id
    ids = [tokenizer.encode_special(t) for t in SPECIAL_TOKENS]
    assert len(set(ids)) == len(SPECIAL_TOKENS)
    assert tokenizer.get_bos_token_id() == tokenizer.encode_special("<|bos|>")
    # specials are NOT special-cased in ordinary text encoding
    ids = tokenizer.encode("<|bos|>")
    assert len(ids) > 1, "special token strings in plain text should not collapse to one token"


def test_encode_prepend_append(tokenizer):
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode("hello", prepend="<|bos|>", append="<|user_end|>")
    assert ids[0] == bos
    assert ids[-1] == tokenizer.encode_special("<|user_end|>")


def test_encode_batch(tokenizer):
    texts = ["hello", "world"]
    ids = tokenizer.encode(texts)
    assert isinstance(ids, list) and len(ids) == 2
    assert ids[0] == tokenizer.encode("hello")


def test_render_conversation_masks(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "bye"},
        {"role": "assistant", "content": "later"},
    ]}
    ids, mask = tokenizer.render_conversation(conversation)
    assert len(ids) == len(mask)
    # first token is bos and is not supervised
    assert ids[0] == tokenizer.get_bos_token_id() and mask[0] == 0
    # supervised tokens are exactly: assistant content + assistant_end tokens
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    supervised_ids = [i for i, m in zip(ids, mask) if m == 1]
    expected = tokenizer.encode("hello!") + [assistant_end] + tokenizer.encode("later") + [assistant_end]
    assert supervised_ids == expected
    # user content tokens are never supervised
    user_start = tokenizer.encode_special("<|user_start|>")
    assert all(m == 0 for i, m in zip(ids, mask) if i == user_start)


def test_render_conversation_system_message_merged(tokenizer):
    without_system = {"messages": [
        {"role": "user", "content": "sys prompt\n\nhi"},
        {"role": "assistant", "content": "yo"},
    ]}
    with_system = {"messages": [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]}
    assert tokenizer.render_conversation(with_system) == tokenizer.render_conversation(without_system)


def test_render_conversation_tool_parts(tokenizer):
    # tool calls are supervised, tool results (come from tool execution) are not
    conversation = {"messages": [
        {"role": "user", "content": "add"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "sure"},
            {"type": "tool_call", "text": '{"name": "python", "code": "1+1"}'},
            {"type": "tool_result", "text": "2"},
            {"type": "text", "text": "it is 2"},
        ]},
    ]}
    ids, mask = tokenizer.render_conversation(conversation)
    tool_call = tokenizer.encode_special("<|tool_call|>")
    tool_result = tokenizer.encode_special("<|tool_result|>")
    tool_result_end = tokenizer.encode_special("<|tool_result_end|>")
    # the tool call and its delimiters are supervised
    assert mask[ids.index(tool_call)] == 1
    # the tool result and its delimiters are not
    start, end = ids.index(tool_result), ids.index(tool_result_end)
    assert all(m == 0 for m in mask[start:end + 1])


def test_render_conversation_thought_parts(tokenizer):
    # thought parts and their delimiters are supervised during CoT SFT
    conversation = {"messages": [
        {"role": "user", "content": "Why is sky blue?"},
        {"role": "assistant", "content": [
            {"type": "thought", "text": "Rayleigh scattering causes shorter blue wavelengths to scatter."},
            {"type": "text", "text": "The sky is blue because of Rayleigh scattering."},
        ]},
    ]}
    ids, mask = tokenizer.render_conversation(conversation)
    thought_start = tokenizer.encode_special("<|thought|>")
    thought_end = tokenizer.encode_special("<|thought_end|>")
    # both the delimiters and the thought tokens must be supervised (mask == 1)
    t_start = ids.index(thought_start)
    t_end = ids.index(thought_end)
    assert all(m == 1 for m in mask[t_start:t_end + 1])


def test_render_conversation_truncation(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hello " * 100},
        {"role": "assistant", "content": "world " * 100},
    ]}
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=32)
    assert len(ids) == 32 and len(mask) == 32


def test_render_for_completion(tokenizer):
    conversation = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "this gets stripped"},
    ]}
    ids = tokenizer.render_for_completion(conversation)
    # ends with assistant_start, primed for a completion
    assert ids[-1] == tokenizer.encode_special("<|assistant_start|>")
    # the assistant response itself must not be present
    stripped = tokenizer.encode("this gets stripped")
    assert not any(ids[i:i + len(stripped)] == stripped for i in range(len(ids)))


def test_safe_chunking_long_non_whitespace(tokenizer):
    from nanochat.tokenizer import _safe_chunk_text, MAX_NO_WHITESPACE_CHARS

    # Test short string -> no chunking
    short_str = "hello world"
    assert _safe_chunk_text(short_str) == [short_str]

    # Test giant 75,000 char non-whitespace string (like giant base64 or minified code)
    giant_str = "a" * 75_000
    chunks = _safe_chunk_text(giant_str)
    assert len(chunks) == 3
    assert all(len(c) <= MAX_NO_WHITESPACE_CHARS for c in chunks)
    assert "".join(chunks) == giant_str

    # Test encode-decode roundtrip on giant non-whitespace string
    encoded = tokenizer.encode(giant_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == giant_str

