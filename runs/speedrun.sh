#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 1.5 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~1.5 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer

# Download initial shards for tokenizer and pretraining
python -m nanochat.dataset --source climbmix -n 8
python -m nanochat.dataset --source indonesian -n 2
# Immediately also kick off downloading more ClimbMix shards in the background while tokenizer trains
python -m nanochat.dataset --source climbmix -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 98304 on ClimbMix + Indonesian corpus
python -m scripts.tok_train --vocab-size=98304
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (trained with 70% ClimbMix + 30% Indonesian)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --indonesian-ratio=0.30 \
    --fp8 \
    --run=$WANDB_RUN

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice, reasoning)

# run SFT across multi-task mixture
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --alpaca-id-epochs=1 \
    --reasoning-epochs=1 \
    --glaive-epochs=1 \
    --gsm8k-epochs=4 \
    --mmlu-epochs=3 \
    --run=$WANDB_RUN

# evaluate the model across global and Indonesian benchmarks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Apa ibu kota negara Indonesia?"
