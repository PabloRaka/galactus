"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Tool registry: maps tool name -> execution function
# Each tool function takes a dict of arguments and returns a string result (or None on failure)

import json

@contextmanager
def timeout(duration, formula):
    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    if has_alarm:
        def timeout_handler(signum, frame):
            raise Exception(f"'{formula}': timed out after {duration} seconds")
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)
    try:
        yield
    finally:
        if has_alarm:
            signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        return None

from nanochat.execution import execute_code

def execute_python(args):
    """
    Execute Python code or expression.
    Fast path: pure math expressions evaluated in-process with timeout.
    General path: multi-line code/print statements evaluated in isolated sandbox.
    """
    expr = args.get("code") or args.get("expression") or args.get("expr") or ""
    if not expr:
        return None

    # Fast path for pure math (e.g. GSM8K calculator)
    cleaned = expr.replace(",", "")
    if all(x in "0123456789*+-/.() " for x in cleaned) and "**" not in cleaned:
        res = eval_with_timeout(cleaned)
        if res is not None:
            return str(res)

    # General sandboxed execution (supports print, loops, functions, variables)
    exec_res = execute_code(expr, timeout=5.0)
    if exec_res.success:
        out = exec_res.stdout.strip()
        return out if out else "(executed successfully, no output)"
    else:
        return f"[error] {exec_res.error or 'execution failed'}"

def execute_bash(args):
    """
    Execute a shell command via subprocess with timeout and output cap.
    Args: {"name": "bash", "command": "ls -la"}
    """
    import subprocess
    command = args.get("command", "")
    if not command:
        return None
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=10,  # ponytail: 10s ceiling, upgrade to configurable if needed
        )
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr] {result.stderr}"
        # Cap output to prevent token explosion during inference
        max_chars = 2000
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... (truncated, {len(output)} total chars)"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "[error] command timed out after 10 seconds"
    except Exception as e:
        return f"[error] {e}"

def execute_web_search(args):
    """
    Search the web using DuckDuckGo instant answers API (no API key needed).
    Args: {"name": "web_search", "query": "python list comprehension"}
    """
    import urllib.request
    import urllib.parse
    query = args.get("query", "")
    if not query:
        return None
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": "1", "skip_disambig": "1"
        })
        req = urllib.request.Request(url, headers={"User-Agent": "nanochat/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        # Build a concise result from the API response
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        if data.get("Answer"):
            parts.append(data["Answer"])
        for topic in (data.get("RelatedTopics") or [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append(f"- {topic['Text']}")
        if not parts:
            return f"No instant results found for: {query}"
        return "\n".join(parts)
    except Exception as e:
        return f"[error] web search failed: {e}"

from nanochat.coding_tools import (
    execute_read_file,
    execute_write_file,
    execute_edit_file,
    execute_grep_search,
    execute_list_files,
)

# ponytail: registry is a plain dict, add new tools by adding entries
TOOL_REGISTRY = {
    "python": execute_python,
    "bash": execute_bash,
    "web_search": execute_web_search,
    "read_file": execute_read_file,
    "write_file": execute_write_file,
    "edit_file": execute_edit_file,
    "grep_search": execute_grep_search,
    "list_files": execute_list_files,
}

def dispatch_tool_call(tool_call_json):
    """Parse a JSON tool call and dispatch to the appropriate tool. Returns result string or None."""
    try:
        call = json.loads(tool_call_json)
    except (json.JSONDecodeError, TypeError):
        return None
    name = call.get("name")
    if name not in TOOL_REGISTRY:
        return None
    return TOOL_REGISTRY[name](call)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

from nanochat.structured import JSONConstraint, extract_and_validate_json

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None, json_constraint=None, max_tool_calls=5):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_tool_call = False # Whether we are inside a tool call block
        self.tool_call_tokens = [] # Tokens of the current tool call payload
        self.tool_call_count = 0 # Number of tool executions in this turn
        self.max_tool_calls = max_tool_calls # Max allowed tool calls before forced stop
        self.tool_call_history = [] # History of (payload, result) for cycle detection
        self.tools_disabled = False # Disabled if max calls or cycle detected
        self.json_constraint = json_constraint # Optional JSONConstraint state tracker
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42,
                 max_tool_calls=5, response_format=None, response_schema=None):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # Allocate the KV cache in the compute dtype so it matches what the forward pass emits
        dtype = COMPUTE_DTYPE
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # JSON mode setup
        is_json_mode = (response_format == {"type": "json_object"} or response_format == "json_object" or response_schema is not None)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        tool_call_tok = get_special("<|tool_call|>")
        tool_call_end_tok = get_special("<|tool_call_end|>")
        tool_result_tok = get_special("<|tool_result|>")
        tool_result_end_tok = get_special("<|tool_result_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample with anti-loop protection
        row_states = [
            RowState(
                tokens.copy(),
                json_constraint=JSONConstraint(schema=response_schema) if is_json_mode else None,
                max_tool_calls=max_tool_calls,
            )
            for _ in range(num_samples)
        ]

        # 4) Main generation loop
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)

                # Decode token to update JSON constraint tracker if active
                if state.json_constraint is not None and not state.in_tool_call:
                    token_str = self.tokenizer.decode([next_token])
                    state.json_constraint.update(token_str)
                    if state.json_constraint.is_completed:
                        state.completed = True

                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool call logic with anti-loop protections
                if next_token == tool_call_tok:
                    state.in_tool_call = True
                    state.tool_call_tokens = []
                elif next_token == tool_call_end_tok and state.in_tool_call:
                    state.in_tool_call = False
                    if state.tool_call_tokens:
                        payload = self.tokenizer.decode(state.tool_call_tokens).strip()
                        state.tool_call_count += 1

                        # Safeguard 1: Tool loop limit exceeded
                        if state.tool_call_count > state.max_tool_calls or state.tools_disabled:
                            state.tools_disabled = True
                            result = f"[error] Maximum tool iteration limit ({state.max_tool_calls}) reached. Please provide your final answer now without calling any more tools."

                        # Safeguard 2: Cycle / Repeated failing call detection
                        elif (
                            len(state.tool_call_history) >= 1
                            and payload == state.tool_call_history[-1][0]
                            and state.tool_call_history[-1][1].startswith("[error]")
                        ):
                            state.tools_disabled = True
                            result = "[error] Repeated identical failing tool call detected. Tool loop aborted to prevent infinite execution. Provide your best answer now."

                        # Normal dispatch
                        else:
                            result = dispatch_tool_call(payload)
                            if result is None:
                                result = "[error] tool call failed or returned no result"
                            state.tool_call_history.append((payload, result))

                        result_tokens = self.tokenizer.encode(result)
                        state.forced_tokens.append(tool_result_tok)
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(tool_result_end_tok)
                    state.tool_call_tokens = []
                elif state.in_tool_call:
                    state.tool_call_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    def run_agent_turn(self, tokens, max_tool_calls=5, max_tokens=512, **kwargs):
        """
        Run a single autonomous agent turn with anti-loop protection and self-correction.
        Returns a dict: {"output_tokens": list[int], "response_text": str}
        """
        prefix_len = len(tokens)
        results, _ = self.generate_batch(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            max_tool_calls=max_tool_calls,
            **kwargs,
        )
        gen_tokens = results[0][prefix_len:]
        decoded = self.tokenizer.decode(gen_tokens)
        return {
            "output_tokens": gen_tokens,
            "response_text": decoded,
        }

    def generate_json(self, tokens, schema=None, **kwargs):
        """
        Generate structured JSON output validated against an optional JSON schema.
        Returns: (parsed_json: Optional[Any], is_valid: bool, error: Optional[str])
        """
        prefix_len = len(tokens)
        results, _ = self.generate_batch(
            tokens,
            num_samples=1,
            response_format={"type": "json_object"},
            response_schema=schema,
            **kwargs,
        )
        gen_tokens = results[0][prefix_len:]
        decoded_text = self.tokenizer.decode(gen_tokens)
        return extract_and_validate_json(decoded_text, schema=schema)



if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
