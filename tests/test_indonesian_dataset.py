import os
import gc
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from nanochat.dataset import parquets_iter_batched, list_parquet_files
from nanochat.dataloader import _document_batches


def create_dummy_parquet(filepath, texts, col_name="text"):
    table = pa.Table.from_arrays([pa.array(texts)], names=[col_name])
    pq.write_table(table, filepath, row_group_size=len(texts) // 2 or 1)


def test_indonesian_dataset_listing_and_iter():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "shard_00000.parquet")
        create_dummy_parquet(f1, ["Indonesia adalah negara kepulauan.", "Bahasa Indonesia adalah bahasa persatuan."])

        files = list_parquet_files(data_dir=tmpdir)
        assert len(files) == 1

        batches = list(parquets_iter_batched(split="train", data_dir=tmpdir))
        assert len(batches) > 0
        all_texts = [t for b in batches for t in b]
        assert any("Indonesia" in t for t in all_texts)
        del batches
        gc.collect()


def test_bilingual_document_batches(monkeypatch):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_climb, \
         tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_indo:

        create_dummy_parquet(os.path.join(tmp_climb, "shard_00000.parquet"), ["ClimbMix English web doc 1", "ClimbMix English web doc 2"] * 10)
        create_dummy_parquet(os.path.join(tmp_indo, "shard_00000.parquet"), ["Artikel ensiklopedia Indonesia 1", "Artikel ensiklopedia Indonesia 2"] * 10)

        monkeypatch.setattr("nanochat.dataloader.list_parquet_files", lambda data_dir=None, **kwargs: (
            [os.path.join(tmp_indo, "shard_00000.parquet")] if data_dir == tmp_indo
            else [os.path.join(tmp_climb, "shard_00000.parquet")]
        ))
        monkeypatch.setattr("nanochat.dataloader.INDONESIAN_DATA_DIR", tmp_indo)

        gen = _document_batches("train", None, tokenizer_batch_size=4, indonesian_ratio=0.5)
        sample_batch, state = next(gen)
        assert len(sample_batch) > 0
        assert isinstance(sample_batch[0], str)

        del gen
        gc.collect()


if __name__ == "__main__":
    test_indonesian_dataset_listing_and_iter()
    print("test_indonesian_dataset_listing_and_iter passed!")
