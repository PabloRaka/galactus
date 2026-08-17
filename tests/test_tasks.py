"""
Test the Task container machinery: slicing views, mixtures, and the
HubDataset parquet wrapper (in-memory, no network).

python -m pytest tests/test_tasks.py -v
"""

import numpy as np
import pyarrow as pa
from tasks.common import Task, TaskMixture, HubDataset, render_mc


class ToyTask(Task):
    """A trivial task: example i is just {'i': i, 'tag': tag}."""

    def __init__(self, n=10, tag="a", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.tag = tag

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index, "tag": self.tag}


def test_task_full():
    task = ToyTask(n=10)
    assert len(task) == 10
    assert task[0] == {"i": 0, "tag": "a"}
    assert task[9] == {"i": 9, "tag": "a"}


def test_task_slicing():
    # a view of [5, 10) has 5 examples and maps logical to physical indices
    task = ToyTask(n=10, start=5, stop=10)
    assert len(task) == 5
    assert task[0]["i"] == 5
    # step slicing uses ceil division for the length
    task = ToyTask(n=10, start=0, stop=10, step=3) # 0, 3, 6, 9
    assert len(task) == 4
    assert [task[i]["i"] for i in range(4)] == [0, 3, 6, 9]


def test_mixture_covers_all_examples_deterministically():
    mixture = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert len(mixture) == 8
    examples = [mixture[i] for i in range(8)]
    # every example appears exactly once
    keys = sorted((ex["tag"], ex["i"]) for ex in examples)
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    # the shuffle is deterministic: a second instance yields the same order
    mixture2 = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert examples == [mixture2[i] for i in range(8)]
    # and the tasks are actually interleaved, not concatenated
    assert [ex["tag"] for ex in examples] != ["a"] * 3 + ["b"] * 5


def test_mixture_oversampling():
    # passing a task twice doubles its examples
    mixture = TaskMixture([ToyTask(n=3), ToyTask(n=3)])
    assert len(mixture) == 6


def test_hub_dataset_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubDataset(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_dataset_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubDataset(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubDataset(table)[0] == {"x": 0}


def test_render_mc_letter_binding():
    query = render_mc("What is 1+1?", ("A", "B"), ("1", "2"))
    # the letter must directly follow '=' with no whitespace, so that the
    # prompt token for "A" matches the assistant's bare "A" response token
    assert "=A\n" in query and "=B\n" in query


def test_parse_glaive_chat():
    from tasks.glaive_tool_calling import _parse_glaive_chat
    system_text = "SYSTEM: You have access to get_weather tool."
    chat_text = """USER: What is the weather in Tokyo?
ASSISTANT: <functioncall> {"name": "get_weather", "arguments": "{\\"city\\": \\"Tokyo\\"}"} <|endoftext|>
FUNCTION RESPONSE: {"temperature": "20C", "condition": "Sunny"}
ASSISTANT: The weather in Tokyo is Sunny and 20C. <|endoftext|>"""

    messages = _parse_glaive_chat(system_text, chat_text)
    assert messages[0]["role"] == "system"
    assert "get_weather" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the weather in Tokyo?"
    assert messages[2]["role"] == "assistant"
    # assistant message content is a list of parts with tool_call, tool_result, text
    parts = messages[2]["content"]
    assert any(p["type"] == "tool_call" and "Tokyo" in p["text"] for p in parts)
    assert any(p["type"] == "tool_result" and "20C" in p["text"] for p in parts)
    assert any(p["type"] == "text" and "Sunny" in p["text"] for p in parts)


def test_parse_thought_content():
    from tasks.deep_reasoning import parse_thought_content
    # Test standard <think>...</think> block
    sample = "<think>\nLet's analyze this step by step.\n1. x = 2\n2. y = 3\n</think>\nThe answer is 5."
    parts = parse_thought_content(sample)
    assert len(parts) == 2
    assert parts[0]["type"] == "thought"
    assert "step by step" in parts[0]["text"]
    assert parts[1]["type"] == "text"
    assert "The answer is 5." in parts[1]["text"]

    # Test plain text without thinking
    plain = "Direct answer without thinking."
    parts_plain = parse_thought_content(plain)
    assert len(parts_plain) == 1
    assert parts_plain[0]["type"] == "text"
    assert parts_plain[0]["text"] == plain


def test_alpaca_indonesian_example_formatting(monkeypatch):
    from tasks.alpaca_indonesian import AlpacaGPT4Indonesian
    import pyarrow as pa

    mock_table = pa.Table.from_pydict({
        "instruction": [
            "Jelaskan apa itu gravitasi.",
            "Terjemahkan kalimat ini ke bahasa Inggris.",
        ],
        "input": [
            "",
            "Saya sedang belajar kecerdasan buatan.",
        ],
        "output": [
            "Gravitasi adalah gaya tarik-menarik antara benda bermassa.",
            "I am learning artificial intelligence.",
        ]
    })
    mock_ds = HubDataset(mock_table)
    monkeypatch.setattr("tasks.alpaca_indonesian.load_hub_dataset", lambda *args, **kwargs: mock_ds)

    task_train = AlpacaGPT4Indonesian(split="train", test_ratio=0.5)
    assert task_train.num_examples() == 1
    ex_train = task_train.get_example(0)
    assert len(ex_train["messages"]) == 2
    assert ex_train["messages"][0]["role"] == "user"
    assert ex_train["messages"][1]["role"] == "assistant"

    task_test = AlpacaGPT4Indonesian(split="test", test_ratio=0.5)
    assert task_test.num_examples() == 1
    ex_test = task_test.get_example(0)
    assert len(ex_test["messages"]) == 2
    assert ex_test["messages"][0]["role"] == "user"
    assert ex_test["messages"][1]["role"] == "assistant"

    all_outputs = [ex_train["messages"][1]["content"], ex_test["messages"][1]["content"]]
    assert any("gaya tarik-menarik" in out for out in all_outputs)
    assert any("artificial intelligence" in out for out in all_outputs)
