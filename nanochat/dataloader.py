"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import random
import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files, INDONESIAN_DATA_DIR


def _document_batches(split, resume_state_dict, tokenizer_batch_size, indonesian_ratio=0.0):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Supports single-source ClimbMix or bilingual mixture with Indonesian corpus.
    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train"
    climb_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(climb_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    climb_paths = climb_paths[:-1] if split == "train" else climb_paths[-1:]

    indo_paths = list_parquet_files(INDONESIAN_DATA_DIR) if indonesian_ratio > 0.0 else []
    if indo_paths:
        indo_paths = indo_paths[:-1] if (split == "train" and len(indo_paths) > 1) else indo_paths

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    # Single-stream generator for a list of parquet files
    def make_stream(paths):
        p_idx = 0
        ep = 1
        while True:
            while p_idx < len(paths):
                filepath = paths[p_idx]
                pf = pq.ParquetFile(filepath)
                rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    col_name = "text" if "text" in rg.column_names else rg.column_names[0]
                    batch = rg.column(col_name).to_pylist()
                    batch = [str(x) for x in batch if x]
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (p_idx, rg_idx, ep)
                    rg_idx += ddp_world_size
                p_idx += 1
            p_idx = 0
            ep += 1

    if not indo_paths or indonesian_ratio <= 0.0:
        # Pure ClimbMix stream with exact resume support
        while True:
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(climb_paths):
                filepath = climb_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size
                    base_idx += 1
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    col_name = "text" if "text" in rg.column_names else rg.column_names[0]
                    batch = rg.column(col_name).to_pylist()
                    batch = [str(x) for x in batch if x]
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                    rg_idx += ddp_world_size
                pq_idx += 1
            first_pass = False
            epoch += 1
    else:
        # Bilingual mixture stream
        climb_gen = make_stream(climb_paths)
        indo_gen = make_stream(indo_paths)
        climb_weight = max(0.0, 1.0 - indonesian_ratio)
        weights = [climb_weight, indonesian_ratio]

        while True:
            chosen_gen = random.choices([climb_gen, indo_gen], weights=weights, k=1)[0]
            yield next(chosen_gen)


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, indonesian_ratio=0.0
):
    """
    BOS-aligned dataloader with Best-Fit Cropping and optional Indonesian bilingual support.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size, indonesian_ratio=indonesian_ratio)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
