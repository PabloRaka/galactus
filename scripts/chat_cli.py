"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
import json
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine, TOOL_REGISTRY
from nanochat.checkpoint_manager import load_model
from nanochat.structured import extract_and_validate_json

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('--no-tools', action='store_true', help='Disable tool use system prompt')
parser.add_argument('--hide-thoughts', action='store_true', help='Hide internal thinking/reasoning process')
parser.add_argument('--json-mode', action='store_true', help='Force valid JSON structured output')
parser.add_argument('--schema', type=str, default=None, help='JSON schema string or path to .json schema file')
parser.add_argument('--tdd', action='store_true', help='Enforce Autonomous Test-Driven Development (Red-Green-Refactor) protocol')
parser.add_argument('--max-tool-calls', type=int, default=5, help='Max tool execution iterations per turn to prevent loops')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# Parse schema if provided
parsed_schema = None
if args.schema:
    args.json_mode = True
    try:
        if args.schema.strip().startswith("{"):
            parsed_schema = json.loads(args.schema)
        else:
            with open(args.schema, "r") as f:
                parsed_schema = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to parse schema: {e}")

from nanochat.tdd import TDD_SYSTEM_PROMPT

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")
thought_tok = tokenizer.encode_special("<|thought|>")
thought_end_tok = tokenizer.encode_special("<|thought_end|>")
tool_call_tok = tokenizer.encode_special("<|tool_call|>")
tool_call_end_tok = tokenizer.encode_special("<|tool_call_end|>")
tool_result_tok = tokenizer.encode_special("<|tool_result|>")
tool_result_end_tok = tokenizer.encode_special("<|tool_result_end|>")

# Build system prompt with available tools
# ponytail: system prompt is prepended to first user message (existing tokenizer behavior)
TOOL_SYSTEM_PROMPT = """You are a helpful software engineering assistant with access to the following tools:

- python: Evaluate Python code/expression. Usage: <|tool_call|>{"name": "python", "code": "code"}<|tool_call_end|>
- bash: Execute a shell command. Usage: <|tool_call|>{"name": "bash", "command": "command"}<|tool_call_end|>
- web_search: Search the web. Usage: <|tool_call|>{"name": "web_search", "query": "query"}<|tool_call_end|>
- read_file: View file contents with line numbers. Usage: <|tool_call|>{"name": "read_file", "path": "path/file.py", "start_line": 1, "end_line": 50}<|tool_call_end|>
- write_file: Create or replace a file. Usage: <|tool_call|>{"name": "write_file", "path": "path/file.py", "content": "..."}<|tool_call_end|>
- edit_file: Replace specific target code with new code. Usage: <|tool_call|>{"name": "edit_file", "path": "path/file.py", "target_content": "old_code", "replacement_content": "new_code"}<|tool_call_end|>
- grep_search: Search for regex/text pattern across codebase. Usage: <|tool_call|>{"name": "grep_search", "query": "pattern", "path": ".", "include": "*.py"}<|tool_call_end|>
- list_files: Explore directory and file trees. Usage: <|tool_call|>{"name": "list_files", "path": ".", "pattern": "*.py"}<|tool_call_end|>

When you use a tool, the result will appear inside <|tool_result|>...<|tool_result_end|> tags. You can use multiple tools in a single response. If a tool returns an error, examine the error message and retry with a corrected call."""

active_system_prompt = TDD_SYSTEM_PROMPT if args.tdd else TOOL_SYSTEM_PROMPT

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode (with Deep Reasoning & Tools)")
print("-" * 50)
if args.tdd:
    print("Mode: 🔴 🟢 🔄 Autonomous Test-Driven Development (TDD Protocol Active)")
if not args.no_tools:
    tool_names = ", ".join(TOOL_REGISTRY.keys())
    print(f"Tools enabled: {tool_names}")
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]
is_first_message = True

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        is_first_message = True
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Inject system prompt into the first user message (tokenizer merges system+user)
    if is_first_message and not args.no_tools:
        full_input = active_system_prompt + "\n\n" + user_input
        is_first_message = False
    else:
        full_input = user_input

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(full_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 512,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "max_tool_calls": args.max_tool_calls,
        "response_format": {"type": "json_object"} if args.json_mode else None,
        "response_schema": parsed_schema,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    in_thought_display = False
    in_tool_call_display = False
    in_tool_result_display = False
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)

        # Visual feedback for thinking / reasoning process
        if token == thought_tok:
            in_thought_display = True
            if not args.hide_thoughts:
                print("\n💭 Thinking:\n", end="", flush=True)
        elif token == thought_end_tok:
            in_thought_display = False
            if not args.hide_thoughts:
                print("\n\n💡 Answer:\n", end="", flush=True)
        # Visual feedback for tool calls
        elif token == tool_call_tok:
            in_tool_call_display = True
            print("\n  🔧 Tool call: ", end="", flush=True)
        elif token == tool_call_end_tok:
            in_tool_call_display = False
            print("", flush=True)
        elif token == tool_result_tok:
            in_tool_result_display = True
            print("  📤 Result: ", end="", flush=True)
        elif token == tool_result_end_tok:
            in_tool_result_display = False
            print("\n", end="", flush=True)
        else:
            if in_thought_display and args.hide_thoughts:
                continue
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()

    # If in JSON mode, validate output structure
    if args.json_mode:
        full_response_text = tokenizer.decode(response_tokens)
        parsed_obj, is_valid, err = extract_and_validate_json(full_response_text, schema=parsed_schema)
        if is_valid:
            print("  ✓ [Schema Verified: Valid JSON]")
        else:
            print(f"  ✗ [Schema Notice: {err}]")

    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
