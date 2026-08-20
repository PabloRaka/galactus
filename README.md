# 🌌 Galactus

**High-Performance Full-Stack LLM Training, Bilingual Indonesian, Polyglot Coding & Deep Reasoning Engine**  
*An advanced, production-grade fork and extension of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.*

---

## 🌟 Overview

**Galactus** is a minimal, ultra-efficient, and hackable full-stack LLM harness designed to train, finetune, evaluate, and deploy high-capability language models from scratch on a budget. 

Building upon the elegant architecture of `nanochat`, Galactus incorporates state-of-the-art enhancements across **Indonesian-optimized BPE tokenization**, **bilingual pretraining mixtures**, **deep reasoning (CoT)**, **autonomous agentic tool calling**, and **multi-hardware acceleration** (including **AMD Instinct MI300X / ROCm** and **NVIDIA Hopper H100 / FA3**).

---

## 🚀 Key Features & Architectural Upgrades

### 1. 🔤 98,304 (~96K) Indonesian & Code-Optimized BPE Tokenizer
- **Indonesian Linguistic Support**: Native regex support for Indonesian hyphenated reduplication (kata ulang: *anak-anak*, *berhari-hari*, *bersama-sama*) and clitics/contractions (*-ku*, *-mu*, *-nya*, *'kan*).
- **Code Literal Preservation**: Hexadecimal (`0xFF00FF`), binary (`0b11010101`), and octal (`0o755`) literals are matched as atomic tokens before 3-digit numeric splitting.
- **Superior Compression**:
  - **Code**: **+40.3% better** compression than GPT-2 (344 tokens vs 576 tokens on benchmarks).
  - **Indonesian**: **+7.3% better** compression than GPT-2 with native morpheme merges.
  - **Science / STEM**: **+16.1% better** compression than GPT-4 (209 tokens vs 249 tokens).
  - **Math**: **+1.7% better** compression than GPT-4.
- **Enterprise Safety Guardrails (Kimi k3 / OpenAI style)**: Text chunking bounded at 400k characters to prevent PyO3 Rust panics and 25k non-whitespace limits to eliminate ReDoS catastrophic backtracking.
- **GPU Matrix-Aligned**: Vocab size of $98,304 = 3 \times 2^{15}$ divides cleanly by 64, 128, and 256 for maximum Tensor Core / Matrix Core throughput.

### 2. 📚 Bilingual Stream Pretraining
- **ClimbMix-400B (General Web, STEM, Math & Code)**: NVIDIA `ClimbMix-400B` curated educational web corpus, GitHub repositories, and scientific papers.
- **Indonesian Knowledge (Wikipedia ID)**: Full Indonesian Wikipedia encyclopedic texts (`20231101.id`) for rich vocabulary and cultural knowledge.
- **Dynamic Sampling**: Configure pretraining bilingual ratios seamlessly:
  ```bash
  python -m scripts.base_train --depth 20 --indonesian-ratio 0.30
  ```

### 3. 🧠 Deep Reasoning & Autonomous Agentic Tool Calling
- **Deep Reasoning (CoT)**: Native `<|thought|>` ... `<|thought_end|>` reasoning tags with supervised loss masking for multi-step thought generation.
- **Autonomous Tool Execution**: Built-in sandboxed tools (`read_file`, `write_file`, `edit_file`, `grep_search`, `list_files`, `python`, `bash`, `web_search`) with automatic JSON repair, schema validation, and anti-infinite-loop safeguards.
- **TDD Protocol**: Optional Test-Driven Development system prompt enforcing Red-Green-Refactor cycles before writing production code.

### 4. ⚡ Multi-Hardware Attention Engine
- **NVIDIA Hopper (H100/H800)**: Direct FlashAttention-3 (`varunneal/flash-attention-3`) integration.
- **AMD Instinct MI300X / ROCm (CDNA3 - gfx942)**: Native FlashAttention-2 ROCm dispatch via Composable Kernel (CK) & PyTorch ROCm SDPA backend (>800 TFLOPS BF16).
- **Universal Fallback**: Automatic sliding-window PyTorch SDPA for CPU, Apple Silicon (MPS), and legacy GPUs.

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

### 1. Download Pretraining Dataset Shards

```bash
# Download ClimbMix shards (English, Science, Code, Math)
python -m nanochat.dataset --source climbmix -n 10

# Download Indonesian Knowledge shards (Wikipedia ID)
python -m nanochat.dataset --source indonesian -n 2
```

### 2. Train Tokenizer

```bash
# Train 98,304-vocab BPE tokenizer from ClimbMix + Indonesian corpus
python -m scripts.tok_train --vocab-size 98304

# Evaluate tokenizer compression ratios against GPT-2 and GPT-4
python -m scripts.tok_eval
```

### 3. Pretrain Base Model

```bash
# Single GPU training (70% ClimbMix + 30% Indonesian)
python -m scripts.base_train --depth 20 --max-seq-len 2048 --device-batch-size 32 --indonesian-ratio 0.30

# Multi-GPU training with torchrun (8x GPU node)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth 24 \
    --max-seq-len 2048 \
    --device-batch-size 16 \
    --indonesian-ratio 0.30 \
    --fp8
```

### 4. Supervised Fine-Tuning (SFT)

Galactus SFT trains conversational, reasoning, and tool calling abilities across a balanced multi-task mixture:
- **General Dialogue**: `HuggingFaceTB/smol-smoltalk` (460K rows)
- **Indonesian Instructions & Chat**: `Ichsan2895/alpaca-gpt4-indonesian` (47.5K rows)
- **Deep Reasoning (CoT)**: `bespokelabs/Bespoke-Stratos-17k` (16.7K rows)
- **Agentic Tool Calling**: `glaiveai/glaive-function-calling-v2` (113K rows)
- **Academic & Logic**: `cais/mmlu` (100K rows) + `openai/gsm8k` (7.5K rows)

```bash
# Run SFT training pipeline
python -m scripts.chat_sft --alpaca-id-epochs 1 --reasoning-epochs 1 --glaive-epochs 1 --gsm8k-epochs 4 --mmlu-epochs 3
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

Run the full automated test suite covering attention engines, tokenization, Indonesian dataset streaming, tool calling, and tasks:

```bash
python -m pytest tests/ -v
```

---

## 📂 Repository Structure

```
galactus/
├── nanochat/                   # Core library
│   ├── checkpoint_manager.py   # Model checkpointing & weights
│   ├── coding_tools.py         # Sandboxed filesystem & coding tools
│   ├── dataloader.py           # Bilingual BOS-aligned best-fit dataloader
│   ├── dataset.py              # Parquet streaming & download dispatcher
│   ├── engine.py               # KV cache, generation state machine & tool dispatcher
│   ├── flash_attention.py      # Multi-hardware attention engine (FA3/FA2/ROCm)
│   ├── gpt.py                  # Transformer architecture (RMSNorm, SwiGLU, RoPE)
│   ├── optim.py                # Muon + AdamW distributed optimizer
│   ├── self_correction.py      # Tool execution & error retry loop
│   ├── structured.py           # JSON schema parser & output validator
│   ├── tdd.py                  # Autonomous TDD protocol & test runner
│   └── tokenizer.py            # 96K BPE tokenizer with Indonesian & code regex
├── scripts/                    # CLI execution entry points
│   ├── base_eval.py            # Pretrain evaluation (BPB / CORE metric)
│   ├── base_train.py           # Pretraining script with bilingual ratio
│   ├── chat_cli.py             # Interactive CLI with CoT & tool feedback
│   ├── chat_eval.py            # Chat & benchmark evaluation (IndoCORE / ChatCORE)
│   ├── chat_sft.py             # Supervised fine-tuning across task mixtures
│   ├── tok_eval.py             # Tokenizer compression benchmark
│   └── tok_train.py            # Bilingual BPE tokenizer training
├── tasks/                      # SFT & evaluation task definitions
│   ├── alpaca_indonesian.py    # Indonesian instruction tuning (Alpaca GPT-4 ID)
│   ├── deep_reasoning.py       # Chain-of-thought mathematical reasoning
│   ├── glaive_tool_calling.py  # Agentic function & tool calling
│   ├── gsm8k.py                # Grade school math problems
│   ├── indo_eval.py            # Indonesian evaluation benchmarks
│   ├── mmlu.py                 # Multi-subject academic knowledge
│   └── smoltalk.py             # Conversational multi-turn dialogues
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
