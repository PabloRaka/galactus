# 🌌 Galactus

**High-Performance Full-Stack LLM Training, Polyglot Coding & Deep Reasoning Engine**  
*An advanced, production-grade fork and extension of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.*

---

## 🌟 Overview

**Galactus** is a minimal, ultra-efficient, and hackable full-stack LLM harness designed to train, finetune, evaluate, and deploy high-capability language models from scratch on a budget. 

Building upon the elegant architecture of `nanochat`, Galactus incorporates state-of-the-art enhancements across tokenization, polyglot data mixtures, deep reasoning, agentic tool execution, and multi-hardware acceleration (including **AMD Instinct MI300X / ROCm** and **NVIDIA Hopper H100 / FA3**).

---

## 🚀 Key Features & Architectural Upgrades

### 1. 🔤 98,304 (~96K) Modern BPE Tokenizer
- **Qwen 2.5 & DeepSeek Split Pattern**: Integrated Han character isolation (`[\p{Han}]+`), 3-digit numeric grouping (`\p{N}{1,3}`), and multilingual Unicode operator boundaries.
- **Superior Compression**:
  - **Code**: **+40.8% better** compression than GPT-2 (341 tokens vs 576 tokens on benchmarks).
  - **Science / STEM**: **+16.1% better** than GPT-4.
  - **ClimbMix & Math**: Outperforms GPT-4 compression on academic texts.
- **Enterprise Safety Guardrails (Kimi k3 / OpenAI style)**: Text chunking bounded at 400k characters to prevent PyO3 Rust panics and 25k non-whitespace limits to eliminate ReDoS catastrophic backtracking.
- **GPU Aligned**: Vocab size of $98,304 = 3 \times 2^{15}$ divides cleanly by 64, 128, and 256 for maximum Tensor Core / Matrix Core throughput.

### 2. 📚 Polyglot Multi-Source Hybrid Pretraining
- **General Web & STEM (`climbmix` - 30%)**: NVIDIA `ClimbMix-400B` curated educational web corpus.
- **Indonesian Knowledge (`indonesian` - 10%)**: Indonesian Wikipedia (`20231101.id`) & C4-ID encyclopedic texts and cultural context.
- **Targeted Polyglot Code (`code` - 30%)**: Clean source code covering:
  - 🟦 TypeScript (`.ts`, `.tsx`), JavaScript (`.js`, `.jsx`)
  - 🐘 PHP (`.php`)
  - ⚙️ C (`.c`, `.h`), C++ (`.cpp`, `.hpp`)
  - 🐍 Python (`.py`), C# (`.cs`), SQL (`.sql`)
- **Mathematics & Logic (`math` - 25%)**: Integrated `open-web-math/open-web-math` (high-grade LaTeX proofs, university theorems, GSM/MATH concepts).
- **Linux Terminal & CLI (`terminal` - 5%)**: Integrated `missvector/linux-commands` (Bash, Zsh, PowerShell, Linux CLI syntax, flags, pipes, and admin scripts).
- **Dynamic Sampling**: Configure pretraining domain mixtures seamlessly:
  ```powershell
  python -m scripts.base_train --web-ratio 0.30 --indonesian-ratio 0.10 --code-ratio 0.30 --math-ratio 0.25 --terminal-ratio 0.05
  ```

### 3. ⚡ Multi-Hardware Attention Engine
- **NVIDIA Hopper (H100/H800)**: Direct FlashAttention-3 (`varunneal/flash-attention-3`) integration.
- **AMD Instinct MI300X / ROCm (CDNA3 - gfx942)**: Native FlashAttention-2 ROCm dispatch via Composable Kernel (CK) & PyTorch ROCm SDPA backend (>800 TFLOPS BF16).
- **Universal Fallback**: Automatic sliding-window PyTorch SDPA for CPU, Apple Silicon (MPS), and legacy GPUs.

### 4. 🧠 Deep Reasoning & Autonomous Tool Calling
- **Deep Reasoning (CoT)**: Native `<|thought|>` ... `<|thought_end|>` reasoning tags with automatic loss masking.
- **Tool Calling Suite**: Built-in sandboxed tools (`python`, `bash`, `read_file`, `write_file`, `edit_file`, `grep_search`, `list_files`, `web_search`) with self-correction feedback loop.
- **JSON Schema Enforcer**: Zero-dependency schema validation and repair for structured output.
- **🔴 🟢 🔄 Autonomous TDD Protocol**: Test-Driven Development system prompt enforcing Red-Green-Refactor cycles before writing production code.

---

## 📦 Quick Start

### Installation

Galactus uses [uv](https://docs.astral.sh/uv/) for fast, deterministic dependency management:

```bash
# Clone the repository
git clone https://github.com/PabloRaka/galactus.git
cd galactus

# Option A: NVIDIA CUDA (A100/H100/RTX)
uv sync --extra gpu --group dev

# Option B: AMD ROCm (Instinct MI300X / MI250X - includes FlashAttention-2)
uv sync --extra rocm --group dev

# Option C: CPU / Apple Silicon (MPS)
uv sync --extra cpu --group dev

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows
```

---

## 🏃 Pipeline Workflow

### 1. Download Pretraining Dataset Sources
```bash
# Download ClimbMix shards (General web)
python -m nanochat.dataset --source climbmix -n 10

# Download Indonesian Knowledge shards (Wikipedia ID)
python -m nanochat.dataset --source indonesian -n 2

# Download Polyglot Code shards (JS, TS, PHP, C, C++, Python, SQL)
python -m nanochat.dataset --source code -n 2

# Download Mathematics & LaTeX shards (OpenWebMath)
python -m nanochat.dataset --source math -n 2

# Download Linux Terminal / Shell shards (Bash, Zsh, CLI commands)
python -m nanochat.dataset --source terminal -n 1
```

### 2. Train Tokenizer
```bash
# Train 98,304-vocab BPE tokenizer (completes in ~12 seconds)
python -m scripts.tok_train --vocab-size 98304
```

### 3. Pretrain Base Model
```bash
# Single GPU training (Default mix: Web 30%, Indo 10%, Code 30%, Math 25%, Terminal 5%)
python -m scripts.base_train --depth=20 --web-ratio=0.30 --indonesian-ratio=0.10 --code-ratio=0.30 --math-ratio=0.25 --terminal-ratio=0.05

# Multi-GPU training with torchrun (8x GPU node)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --web-ratio=0.30 \
    --indonesian-ratio=0.10 \
    --code-ratio=0.30 \
    --math-ratio=0.25 \
    --terminal-ratio=0.05 \
    --fp8
```

### 4. Supervised Fine-Tuning (SFT)
Galactus SFT trains conversational abilities across a balanced multi-task mixture:
- **General Dialogue**: `HuggingFaceTB/smol-smoltalk` (460K rows)
- **Indonesian Instructions & Reasoning**: `Ichsan2895/alpaca-gpt4-indonesian` (50K rows)
- **Deep Reasoning (CoT)**: `DeepReasoning` (17K rows of math step-by-step thinking)
- **Agentic Tool Calling**: `GlaiveToolCalling` (113K rows of multi-turn function calls)
- **Academic & Logic**: `MMLU` (100K rows) + `GSM8K` (8K rows)

```bash
# Run full SFT training pipeline
python -m scripts.chat_sft --alpaca-id-epochs 1 --reasoning-epochs 1 --glaive-epochs 1
```

### 5. Interactive Chat CLI (with Reasoning & Tools)
```bash
# Standard conversational mode
python -m scripts.chat_cli

# Autonomous Test-Driven Development (TDD) mode
python -m scripts.chat_cli --tdd

# Hide internal thinking process
python -m scripts.chat_cli --hide-thoughts
```

### 6. Benchmark Evaluation Suite
Evaluate Galactus across standard global benchmarks and native Indonesian benchmarks:
```bash
# Evaluate Indonesian benchmarks (IndoMMLU + IndoReasoning -> IndoCORE metric)
python -m scripts.chat_eval -i sft -a "IndoMMLU|IndoReasoning"

# Evaluate full benchmark suite (Global ChatCORE + Indonesian IndoCORE)
python -m scripts.chat_eval -i sft
```

---

## 🧪 Running Unit Tests

Run the full automated test suite (74+ tests covering attention, tokenization, multi-source streaming, TDD, and tools):

```bash
uv run --with pytest pytest tests/ -v
```

---

## 📂 Repository Structure

```
galactus/
├── nanochat/                   # Core library
│   ├── checkpoint_manager.py   # Model checkpointing & weights
│   ├── coding_tools.py         # Sandboxed filesystem & coding tools
│   ├── dataloader.py           # Multi-source BOS-aligned best-fit dataloader
│   ├── dataset.py              # Parquet streaming & download dispatcher
│   ├── engine.py               # KV cache & generation state machine
│   ├── flash_attention.py      # Multi-hardware attention engine (FA3/FA2/ROCm)
│   ├── gpt.py                  # Transformer architecture (RMSNorm, SwiGLU, RoPE)
│   ├── optim.py                # Muon + AdamW distributed optimizer
│   ├── self_correction.py      # Tool execution & error retry loop
│   ├── structured.py           # JSON schema parser & output validator
│   ├── tdd.py                  # Autonomous TDD protocol & test runner
│   └── tokenizer.py            # 96K BPE tokenizer with regex split & guardrails
├── scripts/                    # CLI execution entry points
│   ├── base_train.py           # Pretraining script with domain weighting
│   ├── chat_cli.py             # Interactive CLI with CoT & tool feedback
│   ├── chat_sft.py             # Supervised fine-tuning across task mixtures
│   ├── tok_train.py            # Tokenizer BPE training
│   └── ...
├── tasks/                      # SFT & evaluation task definitions
│   ├── deep_reasoning.py       # Chain-of-thought mathematical reasoning
│   ├── glaive_tool_calling.py  # Agentic function calling
│   ├── smoltalk.py             # Conversational multi-turn dialogues
│   └── ...
├── tests/                      # Comprehensive pytest test suite
└── pyproject.toml              # Project configuration & dependencies
```

---

## 🙏 Acknowledgements & Attribution

Galactus is built upon the foundation of **[nanochat](https://github.com/karpathy/nanochat)** by **Andrej Karpathy**. We express our deep appreciation to:
- **Andrej Karpathy** for creating `nanochat` and `nanoGPT`.
- **Keller Jordan** and the `modded-nanogpt` community for compute-optimal training ideas.
- **HuggingFace** for `smollm-corpus`, `smoltalk`, and dataset hosting.
- **Dao-AILab** and **AMD ROCm team** for FlashAttention innovations.
- The open-source AI and research community for making frontier LLM development accessible.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
