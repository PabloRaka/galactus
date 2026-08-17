"""
Glaive Function Calling v2 dataset for tool use SFT.
https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2

112K examples of multi-turn conversations with function/tool calls.
Format: system prompt with tool definitions, chat string with USER/ASSISTANT/FUNCTION RESPONSE turns.

We convert to our generic tool_call/tool_result format.
"""

import re
import json
from tasks.common import Task, load_hub_dataset


def _parse_glaive_chat(system_text, chat_text):
    """
    Parse Glaive's flat string format into our messages list.
    Glaive format:
      SYSTEM: ... (in the system column)
      USER: ...
      ASSISTANT: <functioncall> {"name": ..., "arguments": ...} <|endoftext|>
      FUNCTION RESPONSE: {"result": ...}
      ASSISTANT: final text <|endoftext|>
    """
    messages = []

    # Extract system prompt (strip "SYSTEM: " prefix if present)
    if system_text:
        sys_content = system_text.strip()
        if sys_content.startswith("SYSTEM:"):
            sys_content = sys_content[len("SYSTEM:"):].strip()
        if sys_content:
            messages.append({"role": "system", "content": sys_content})

    # Split chat into turns using the role prefixes
    # Pattern: match USER:, ASSISTANT:, or FUNCTION RESPONSE: at line starts
    turns = re.split(r'\n*(?=USER:|ASSISTANT:|FUNCTION RESPONSE:)', chat_text.strip())
    turns = [t.strip() for t in turns if t.strip()]

    i = 0
    while i < len(turns):
        turn = turns[i]

        if turn.startswith("USER:"):
            content = turn[len("USER:"):].strip()
            messages.append({"role": "user", "content": content})
            i += 1

        elif turn.startswith("ASSISTANT:"):
            content = turn[len("ASSISTANT:"):].strip()
            # Remove trailing <|endoftext|>
            content = content.replace("<|endoftext|>", "").strip()

            # Check if this contains a function call
            fc_match = re.search(r'<functioncall>\s*(\{.*\})', content, re.DOTALL)
            if fc_match:
                # This is a tool call turn. Parse the function call JSON.
                fc_json_str = fc_match.group(1).strip()
                # Text before the function call (if any)
                pre_text = content[:fc_match.start()].strip()

                parts = []
                if pre_text:
                    parts.append({"type": "text", "text": pre_text})

                # Parse and re-format as our tool_call format
                fc = None
                try:
                    fc = json.loads(fc_json_str)
                except Exception:
                    try:
                        import ast
                        fc = ast.literal_eval(fc_json_str)
                    except Exception:
                        from nanochat.structured import repair_json
                        try:
                            fc = json.loads(repair_json(fc_json_str))
                        except Exception:
                            fc = None

                if fc is not None and isinstance(fc, dict):
                    # Glaive uses "arguments" (sometimes as string, sometimes as dict)
                    tool_name = fc.get("name", "unknown")
                    tool_args = fc.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except Exception:
                            try:
                                import ast
                                tool_args = ast.literal_eval(tool_args)
                            except Exception:
                                tool_args = {"input": tool_args}
                    if not isinstance(tool_args, dict):
                        tool_args = {"input": tool_args}
                    # Build our tool_call payload
                    our_call = {"name": tool_name}
                    our_call.update(tool_args)
                    parts.append({"type": "tool_call", "text": json.dumps(our_call)})
                else:
                    # Can't parse the function call, treat as plain text
                    parts.append({"type": "text", "text": content})
                    messages.append({"role": "assistant", "content": parts})
                    i += 1
                    continue

                # Look ahead for FUNCTION RESPONSE
                if i + 1 < len(turns) and turns[i + 1].startswith("FUNCTION RESPONSE:"):
                    fr_content = turns[i + 1][len("FUNCTION RESPONSE:"):].strip()
                    parts.append({"type": "tool_result", "text": fr_content})
                    i += 1  # skip the function response turn

                # Look ahead for the assistant's follow-up text
                if i + 1 < len(turns) and turns[i + 1].startswith("ASSISTANT:"):
                    followup = turns[i + 1][len("ASSISTANT:"):].strip()
                    followup = followup.replace("<|endoftext|>", "").strip()
                    if followup:
                        parts.append({"type": "text", "text": followup})
                    i += 1  # skip the followup turn

                messages.append({"role": "assistant", "content": parts})
            else:
                # Plain text response
                messages.append({"role": "assistant", "content": content})

            i += 1

        elif turn.startswith("FUNCTION RESPONSE:"):
            # Orphan function response (shouldn't happen normally), skip
            i += 1
        else:
            i += 1

    return messages


class GlaiveToolCalling(Task):
    """
    Glaive Function Calling v2 dataset.
    113K examples of multi-turn conversations with tool use.
    """

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "GlaiveToolCalling only has train split"
        self.ds = load_hub_dataset("glaiveai/glaive-function-calling-v2", split=split).shuffle(seed=42)

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        system_text = row.get("system", "")
        chat_text = row.get("chat", "")

        messages = _parse_glaive_chat(system_text, chat_text)

        # Sanity: must have at least a user and assistant message
        non_system = [m for m in messages if m["role"] != "system"]
        if len(non_system) < 2:
            # Fallback: return a minimal conversation
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "Hello! How can I help you?"},
            ]

        # Ensure alternating user/assistant after optional system
        # (Glaive data is mostly clean but some edge cases exist)
        cleaned = []
        for m in messages:
            if m["role"] == "system":
                if not cleaned:  # system only at the start
                    cleaned.append(m)
            elif m["role"] == "user":
                if cleaned and cleaned[-1]["role"] == "user":
                    # Merge consecutive user messages
                    cleaned[-1]["content"] += "\n" + m["content"]
                else:
                    cleaned.append(m)
            elif m["role"] == "assistant":
                if not cleaned or cleaned[-1]["role"] == "system":
                    # Assistant cannot be the first message (or right after system without user)
                    continue
                elif cleaned[-1]["role"] == "assistant":
                    # Merge consecutive assistant messages
                    prev = cleaned[-1]["content"]
                    curr = m["content"]
                    if isinstance(prev, str) and isinstance(curr, str):
                        cleaned[-1]["content"] = prev + "\n" + curr
                    elif isinstance(prev, list) and isinstance(curr, list):
                        cleaned[-1]["content"] = prev + curr
                    elif isinstance(prev, str) and isinstance(curr, list):
                        cleaned[-1]["content"] = [{"type": "text", "text": prev}] + curr
                    elif isinstance(prev, list) and isinstance(curr, str):
                        cleaned[-1]["content"] = prev + [{"type": "text", "text": curr}]
                else:
                    cleaned.append(m)

        # Final sanity check: must have at least one user and one assistant
        has_user = any(m["role"] == "user" for m in cleaned)
        has_assistant = any(m["role"] == "assistant" for m in cleaned)
        if not (has_user and has_assistant):
            cleaned = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello! How can I help you today?"},
            ]

        return {"messages": cleaned}
