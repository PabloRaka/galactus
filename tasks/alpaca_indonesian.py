"""
Alpaca GPT-4 Indonesian dataset for Supervised Fine-Tuning (SFT).
https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian

Contains ~50K high-quality Indonesian instruction-following and conversation pairs.
"""

from tasks.common import Task, load_hub_dataset


class AlpacaGPT4Indonesian(Task):
    """
    Alpaca GPT-4 Indonesian dataset.
    50K instruction-response pairs translated and curated for Indonesian language tasks.
    """

    def __init__(self, split="train", test_ratio=0.05, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test", "val"], "Split must be train|test|val"
        # The hub dataset only has a train split, so we split deterministically
        full_ds = load_hub_dataset("Ichsan2895/alpaca-gpt4-indonesian", split="train").shuffle(seed=42)
        total_len = len(full_ds)
        if total_len > 1:
            test_size = max(1, int(total_len * test_ratio))
        else:
            test_size = 1 if split in ["test", "val"] else 0
        train_size = total_len - test_size

        if split in ["test", "val"]:
            self.ds = full_ds
            self.indices = list(range(train_size, total_len))
        else:
            self.ds = full_ds
            self.indices = list(range(0, train_size))

        self.length = len(self.indices)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        real_idx = self.indices[index]
        row = self.ds[real_idx]

        instruction = str(row.get("instruction") or "").strip()
        user_input = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()

        if user_input:
            user_content = f"{instruction}\n\nInput:\n{user_input}"
        else:
            user_content = instruction

        conversation = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        }
        return conversation
