"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# Multi-source pretraining configuration (Polyglot Code, Math, and General Web/STEM)
PRETRAIN_SOURCES = {
    "climbmix": {
        "url_pattern": f"{BASE_URL}/shard_{{:05d}}.parquet",
        "dir": DATA_DIR,
        "max_shards": MAX_SHARD,
        "description": "Curated general web, STEM, encyclopedic, and educational text (ClimbMix-400B)",
    },
    "math": {
        "api_url": "https://huggingface.co/api/datasets/open-web-math/open-web-math/parquet/default/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "math"),
        "description": "High-grade mathematical proofs, formulas, LaTeX derivations (OpenWebMath)",
    },
    "code": {
        "api_url": "https://huggingface.co/api/datasets/codeparrot/github-code-clean/parquet/all-all/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "code"),
        "description": "Polyglot clean source code: JS, TS, PHP, C, C++, Python, C#, SQL (GitHub Code Clean)",
    },
    "terminal": {
        "api_url": "https://huggingface.co/api/datasets/missvector/linux-commands/parquet/default/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "terminal"),
        "description": "Linux terminal, Bash, Zsh, PowerShell, CLI commands and shell scripts (linux-commands)",
    },
    "indonesian": {
        "api_url": "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.id/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "indonesian"),
        "description": "Indonesian encyclopedic, cultural, and knowledge text (Wikipedia ID / C4-ID)",
    },
    "indonesian_instruct": {
        "api_url": "https://huggingface.co/api/datasets/FreedomIntelligence/evol-instruct-indonesian/parquet/default/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "indonesian_instruct"),
        "description": "Complex multi-turn instruction & reasoning in Indonesian (Evol-Instruct-ID)",
    },
    "code_instruct": {
        "api_url": "https://huggingface.co/api/datasets/m-a-p/CodeFeedback-Filtered-Instruction/parquet/default/train",
        "dir": os.path.join(base_dir, "pretrain_sources", "code_instruct"),
        "description": "Multilingual programming instructions (CodeFeedback)",
    },
}


def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()
        # attempt a fallback to the legacy data directory
        data_dir = os.path.join(base_dir, "base_data")

    if not os.path.exists(data_dir):
        return []

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split="train", start=0, step=1, data_dir=None):
    """
    Iterate through a single dataset directory, yielding batches of text from row groups.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir)
    if not parquet_paths:
        return

    if split == "train":
        active_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
    else:
        active_paths = parquet_paths[-1:]

    target_candidate_cols = ["code", "text", "content", "output", "prompt"]

    for filepath in active_paths:
        try:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                # Find appropriate text/code column
                col_name = None
                for c in target_candidate_cols:
                    if c in rg.column_names:
                        col_name = c
                        break
                if col_name is None:
                    col_name = rg.column_names[0]

                texts = rg.column(col_name).to_pylist()
                # Filter non-string or empty items
                texts = [str(t) for t in texts if t]
                if texts:
                    yield texts
        except Exception as e:
            print(f"Warning: skipping corrupt/unreadable shard {filepath}: {e}")


def parquets_iter_multi_source(domain_weights=None, split="train", start=0, step=1):
    """
    Multi-source hybrid pretraining iterator with custom domain balancing.

    Args:
        domain_weights: dict of {source_name: weight}, e.g. {"climbmix": 0.40, "code": 0.35, "math": 0.25}.
                        If None, defaults to pure climbmix (100%).
        split: "train" or "val"
        start, step: DDP rank and world_size for interleaved distribution.

    Yields:
        list[str]: batches of text documents from the weighted domain streams.
    """
    import random

    if domain_weights is None:
        domain_weights = {"climbmix": 1.0}

    # Normalize weights
    total_w = sum(domain_weights.values())
    if total_w <= 0:
        domain_weights = {"climbmix": 1.0}
        total_w = 1.0
    norm_weights = {k: v / total_w for k, v in domain_weights.items() if v > 0}

    # Build active iterators for available sources
    sources = []
    weights = []

    for name, weight in norm_weights.items():
        src_dir = PRETRAIN_SOURCES.get(name, {}).get("dir", DATA_DIR)
        files = list_parquet_files(src_dir)
        if files:
            sources.append((name, src_dir))
            weights.append(weight)

    # If no secondary sources have parquet files on disk, fall back to pure ClimbMix
    if not sources:
        files = list_parquet_files(DATA_DIR)
        if files:
            sources.append(("climbmix", DATA_DIR))
            weights.append(1.0)
        else:
            return

    # Re-normalize active weights
    sum_w = sum(weights)
    weights = [w / sum_w for w in weights]

    # Create infinite generators per active domain
    def create_infinite_source_iter(src_dir):
        while True:
            for batch in parquets_iter_batched(split=split, start=start, step=step, data_dir=src_dir):
                yield batch

    iters = [create_infinite_source_iter(src_dir) for _, src_dir in sources]

    # Weighted streaming loop
    while True:
        # Choose which domain to pull from according to weights
        choice_idx = random.choices(range(len(sources)), weights=weights, k=1)[0]
        try:
            batch = next(iters[choice_idx])
            if batch:
                yield batch
        except StopIteration:
            # Recreate iterator if exhausted
            iters[choice_idx] = create_infinite_source_iter(sources[choice_idx][1])
            batch = next(iters[choice_idx])
            if batch:
                yield batch

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_hf_dataset_source(source_name, num_shards=1):
    """Downloads shards for a secondary pretraining source (e.g. math, code)."""
    import urllib.request
    import json

    cfg = PRETRAIN_SOURCES.get(source_name)
    if not cfg:
        print(f"Unknown source: {source_name}. Available: {list(PRETRAIN_SOURCES.keys())}")
        return

    out_dir = cfg["dir"]
    os.makedirs(out_dir, exist_ok=True)

    if "api_url" in cfg:
        print(f"Fetching shard listing from HuggingFace for '{source_name}'...")
        req = urllib.request.Request(cfg["api_url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            shard_urls = json.loads(r.read())

        target_urls = shard_urls if num_shards == -1 else shard_urls[:num_shards]
        print(f"Downloading {len(target_urls)} shards to {out_dir}...")
        for i, url in enumerate(target_urls):
            filename = f"shard_{i:05d}.parquet"
            dest = os.path.join(out_dir, filename)
            if os.path.exists(dest):
                print(f"  [{i+1}/{len(target_urls)}] Skipping {filename} (already exists)")
                continue
            print(f"  [{i+1}/{len(target_urls)}] Downloading {filename}...")
            req_file = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_file) as resp, open(dest + ".tmp", "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.rename(dest + ".tmp", dest)
        print(f"[OK] Source '{source_name}' downloaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-s", "--source", type=str, default="climbmix", choices=list(PRETRAIN_SOURCES.keys()), help="Dataset source to download")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = all")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    if args.source != "climbmix":
        download_hf_dataset_source(args.source, num_shards=args.num_files)
    else:
        # Prepare the output directory for climbmix
        os.makedirs(DATA_DIR, exist_ok=True)

        num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
        ids_to_download = list(range(num_train_shards))
        ids_to_download.append(MAX_SHARD) # always download the validation shard

        # Download the shards
        print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
        print(f"Target directory: {DATA_DIR}")
        print()
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(download_single_file, ids_to_download)

        # Report results
        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")

