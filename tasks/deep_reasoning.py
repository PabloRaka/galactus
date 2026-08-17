"""
Deep Reasoning / Chain-of-Thought (CoT) dataset for SFT.
Follows the DeepSeek-R1 / OpenAI o-series paradigm of explicit thinking traces.

Default dataset: bespoke-labs/Bespoke-Stratos-17k (17K distilled reasoning examples)
Alternative: open-r1/OpenR1-Math-220k

Parses <think>...</think> tags and maps them to our structured
{"type": "thought", "text": "..."} and {"type": "text", "text": "..."} parts.
"""

import re
from tasks.common import Task, load_hub_dataset


THINK_PATTERNS = [
    re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL),
    re.compile(r"<think>(.*?)</think>", re.DOTALL),
    re.compile(r"\[THOUGHT\](.*?)\[/THOUGHT\]", re.DOTALL),
]


def parse_thought_content(text):
    """
    Parse an assistant message string containing thought traces into parts.
    Handles <|begin_of_thought|>...<|end_of_thought|> (Bespoke-Stratos / Nemotron),
    <think>...</think> (DeepSeek-R1 / Open-R1), and [THOUGHT]...[/THOUGHT].
    Returns a list of parts: [{"type": "thought", "text": ...}, {"type": "text", "text": ...}]
    or [{"type": "text", "text": text}] if no think tags are found.
    """
    if not isinstance(text, str):
        return text

    def clean_answer(ans):
        # Strip solution and answer tags if present
        ans = re.sub(r"</?answer>", "", ans)
        ans = re.sub(r"<\|begin_of_solution\|>|<\|end_of_solution\|>", "", ans)
        ans = re.sub(r"<\|begin_of_thought\|>|<\|end_of_thought\|>", "", ans)
        ans = re.sub(r"</?think>", "", ans)
        return ans.strip()

    for pat in THINK_PATTERNS:
        match = pat.search(text)
        if match:
            thought_text = match.group(1).strip()
            answer_text = clean_answer(text[match.end():])
            parts = []
            if thought_text:
                parts.append({"type": "thought", "text": thought_text})
            if answer_text:
                parts.append({"type": "text", "text": answer_text})
            elif not thought_text:
                parts.append({"type": "text", "text": text})
            return parts if parts else [{"type": "text", "text": text}]

    # Also handle lone closing tags without opening tag
    for close_tag in ["</think>", "<|end_of_thought|>", "[/THOUGHT]"]:
        if close_tag in text:
            thought_text, answer_text = text.split(close_tag, 1)
            for open_tag in ["<think>", "<|begin_of_thought|>", "[THOUGHT]"]:
                thought_text = thought_text.replace(open_tag, "")
            thought_text = thought_text.strip()
            answer_text = clean_answer(answer_text)
            parts = []
            if thought_text:
                parts.append({"type": "thought", "text": thought_text})
            if answer_text:
                parts.append({"type": "text", "text": answer_text})
            return parts if parts else [{"type": "text", "text": text}]

    # Plain text without thinking tags
    clean_text = clean_answer(text)
    return [{"type": "text", "text": clean_text}]


class DeepReasoning(Task):
    """
    Deep reasoning dataset with Chain-of-Thought (CoT) traces.
    Default: bespokelabs/Bespoke-Stratos-17k (17K examples of complex reasoning distilled from DeepSeek-R1).
    """

    def __init__(self, dataset_name="bespokelabs/Bespoke-Stratos-17k", split="train", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        try:
            self.ds = load_hub_dataset(dataset_name, split=split).shuffle(seed=42)
        except Exception:
            # Fallback identifier in case of alternative org naming on HF Hub
            alt_name = "bespoke-labs/Bespoke-Stratos-17k" if dataset_name == "bespokelabs/Bespoke-Stratos-17k" else "bespokelabs/Bespoke-Stratos-17k"
            self.ds = load_hub_dataset(alt_name, split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        # Handle different column conventions across HuggingFace reasoning datasets:
        # 1) "conversations" format (ShareGPT style: [{"from": "human/gpt", "value": "..."}])
        # 2) "messages" format (OpenAI style: [{"role": "user/assistant", "content": "..."}])
        # 3) "prompt" / "response" or "system" / "problem" / "solution"
        messages = []

        if "conversations" in row and isinstance(row["conversations"], list):
            for turn in row["conversations"]:
                sender = turn.get("from", "")
                role = "user" if sender in ["human", "user"] else "assistant"
                val = turn.get("value", "")
                if role == "assistant":
                    content = parse_thought_content(val)
                else:
                    content = val
                messages.append({"role": role, "content": content})

        elif "messages" in row and isinstance(row["messages"], list):
            for turn in row["messages"]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "assistant":
                    content = parse_thought_content(content)
                messages.append({"role": role, "content": content})

        elif "prompt" in row and ("response" in row or "solution" in row):
            prompt = row.get("prompt", "")
            response = row.get("response") or row.get("solution") or ""
            system = row.get("system", "")
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": parse_thought_content(response)})

        elif "problem" in row and "solution" in row:
            problem = row.get("problem", "")
            solution = row.get("solution", "")
            messages.append({"role": "user", "content": problem})
            messages.append({"role": "assistant", "content": parse_thought_content(solution)})

        else:
            # Fallback
            messages = [
                {"role": "user", "content": str(row.get("input", "Hello"))},
                {"role": "assistant", "content": [{"type": "text", "text": str(row.get("output", "Hello!"))}]},
            ]

        # Clean and ensure strict alternating format for tokenizer
        cleaned = []
        for m in messages:
            if m["role"] == "system":
                if not cleaned:
                    cleaned.append(m)
            elif m["role"] == "user":
                if cleaned and cleaned[-1]["role"] == "user":
                    cleaned[-1]["content"] += "\n" + str(m["content"])
                else:
                    cleaned.append(m)
            elif m["role"] == "assistant":
                if not cleaned or cleaned[-1]["role"] == "system":
                    continue
                elif cleaned[-1]["role"] == "assistant":
                    prev = cleaned[-1]["content"]
                    curr = m["content"]
                    if isinstance(prev, list) and isinstance(curr, list):
                        cleaned[-1]["content"] = prev + curr
                    elif isinstance(prev, str) and isinstance(curr, list):
                        cleaned[-1]["content"] = [{"type": "text", "text": prev}] + curr
                    elif isinstance(prev, list) and isinstance(curr, str):
                        cleaned[-1]["content"] = prev + [{"type": "text", "text": curr}]
                    else:
                        cleaned[-1]["content"] = str(prev) + "\n" + str(curr)
                else:
                    cleaned.append(m)

        if len(cleaned) < 2 or not any(m["role"] == "user" for m in cleaned) or not any(m["role"] == "assistant" for m in cleaned):
            cleaned = [
                {"role": "user", "content": "Explain step by step."},
                {"role": "assistant", "content": [{"type": "thought", "text": "Let me think."}, {"type": "text", "text": "Here is the answer."}]},
            ]

        return {"messages": cleaned}
