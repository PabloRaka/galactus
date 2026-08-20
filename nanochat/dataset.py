"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- downloading the files on demand if they are not on disk (ClimbMix & Indonesian corpus)

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import json
import urllib.request
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Pretraining dataset configurations (ClimbMix base + Indonesian corpus)

# ClimbMix-400B
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# Indonesian Corpus (Wikipedia ID & Encyclopedic Text)
INDONESIAN_API_URL = "https://huggingface.co/api/datasets/wikimedia/wikipedia/parquet/20231101.id/train"
INDONESIAN_DATA_DIR = os.path.join(base_dir, "base_data_indonesian")

# -----------------------------------------------------------------------------
# Dataset utility functions

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    if not os.path.exists(data_dir):
        if warn_on_legacy and data_dir == DATA_DIR:
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
        else:
            return []

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
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
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

    for filepath in active_paths:
        try:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                col_name = "text" if "text" in rg.column_names else rg.column_names[0]
                texts = rg.column(col_name).to_pylist()
                texts = [str(t) for t in texts if t]
                if texts:
                    yield texts
        except Exception as e:
            print(f"Warning: skipping corrupt/unreadable shard {filepath}: {e}")


# -----------------------------------------------------------------------------
# Download functions

def download_single_file(index):
    """ Downloads a single ClimbMix file index, with some backoff """
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_indonesian_dataset(num_shards=-1):
    """ Downloads parquet shards for the Indonesian corpus from HuggingFace """
    os.makedirs(INDONESIAN_DATA_DIR, exist_ok=True)
    print(f"Fetching Indonesian dataset listing from HuggingFace...")
    req = urllib.request.Request(INDONESIAN_API_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        shard_urls = json.loads(r.read().decode())

    target_urls = shard_urls if num_shards == -1 else shard_urls[:num_shards]
    print(f"Downloading {len(target_urls)} Indonesian shards to {INDONESIAN_DATA_DIR}...")
    for i, url in enumerate(target_urls):
        filename = f"shard_{i:05d}.parquet"
        dest = os.path.join(INDONESIAN_DATA_DIR, filename)
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
    print(f"[OK] Indonesian dataset downloaded successfully to {INDONESIAN_DATA_DIR}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-s", "--source", type=str, default="climbmix", choices=["climbmix", "indonesian"], help="Dataset source: climbmix or indonesian")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers for climbmix (default: 4)")
    args = parser.parse_args()

    if args.source == "indonesian":
        download_indonesian_dataset(num_shards=args.num_files)
    else:
        # Prepare the output directory for ClimbMix
        os.makedirs(DATA_DIR, exist_ok=True)

        num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
        ids_to_download = list(range(num_train_shards))
        ids_to_download.append(MAX_SHARD)  # always download the validation shard

        print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
        print(f"Target directory: {DATA_DIR}")
        print()
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(download_single_file, ids_to_download)

        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
