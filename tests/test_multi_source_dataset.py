import os
import gc
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nanochat.dataset import parquets_iter_batched, parquets_iter_multi_source
from nanochat.dataloader import _document_batches
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer


def create_dummy_parquet(filepath, texts, col_name="text"):
    table = pa.Table.from_arrays([pa.array(texts)], names=[col_name])
    pq.write_table(table, filepath, row_group_size=len(texts) // 2 or 1)


def test_parquets_iter_batched_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "shard_00000.parquet")
        f2 = os.path.join(tmpdir, "shard_00001.parquet")
        create_dummy_parquet(f1, ["doc A1", "doc A2", "doc A3", "doc A4"])
        create_dummy_parquet(f2, ["doc B1", "doc B2", "doc B3", "doc B4"])

        batches = list(parquets_iter_batched("train", data_dir=tmpdir))
        assert len(batches) > 0
        all_texts = [t for b in batches for t in b]
        assert "doc A1" in all_texts
        del batches
        gc.collect()


def test_parquets_iter_polyglot_code_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "shard_00000.parquet")
        polyglot_samples = [
            "function add(a: number, b: number): number { return a + b; }", # TS
            "<?php echo 'Hello World'; ?>",                                   # PHP
            "def calculate(x: int) -> int: return x * 2",                     # Python
            "SELECT * FROM users WHERE active = 1;",                          # SQL
        ]
        create_dummy_parquet(f, polyglot_samples, col_name="code")

        batches = list(parquets_iter_batched("train", data_dir=tmpdir))
        assert len(batches) > 0
        all_texts = [t for b in batches for t in b]
        assert any("<?php" in s for s in all_texts)
        assert any("def calculate" in s for s in all_texts)
        assert any("SELECT * FROM" in s for s in all_texts)
        del batches
        gc.collect()


def test_parquets_iter_multi_source_weighting(monkeypatch):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_climb, \
         tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_code, \
         tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_math:

        create_dummy_parquet(os.path.join(tmp_climb, "shard_00000.parquet"), ["climb_1", "climb_2"] * 10)
        create_dummy_parquet(os.path.join(tmp_code, "shard_00000.parquet"), ["const x: number = 42;", "def python_app(): pass"] * 10, col_name="code")
        create_dummy_parquet(os.path.join(tmp_math, "shard_00000.parquet"), ["\\int_0^1 x dx = 1/2", "\\sum_{i=1}^n i"] * 10)

        test_sources = {
            "climbmix": {"dir": tmp_climb},
            "code": {"dir": tmp_code},
            "math": {"dir": tmp_math},
        }
        monkeypatch.setattr("nanochat.dataset.PRETRAIN_SOURCES", test_sources)

        weights = {"climbmix": 0.50, "code": 0.30, "math": 0.20}
        gen = parquets_iter_multi_source(domain_weights=weights, split="train")

        collected = []
        for _ in range(100):
            batch = next(gen)
            collected.extend(batch)

        assert len(collected) > 0
        has_climb = any("climb" in s for s in collected)
        has_code = any("const x" in s or "rust" in s for s in collected)
        has_math = any("\\int" in s or "\\sum" in s for s in collected)
        assert has_climb and has_code and has_math

        del gen
        del collected
        gc.collect()


def test_document_batches_multi_source(monkeypatch):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_climb, \
         tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_code:

        create_dummy_parquet(os.path.join(tmp_climb, "shard_00000.parquet"), ["web document 1", "web document 2"] * 10)
        create_dummy_parquet(os.path.join(tmp_code, "shard_00000.parquet"), ["function compute(): void {}", "pub struct User;"] * 10, col_name="code")

        test_sources = {
            "climbmix": {"dir": tmp_climb},
            "code": {"dir": tmp_code},
        }
        monkeypatch.setattr("nanochat.dataset.PRETRAIN_SOURCES", test_sources)
        monkeypatch.setattr("nanochat.dataloader.PRETRAIN_SOURCES", test_sources)

        gen = _document_batches("train", None, tokenizer_batch_size=4, domain_weights={"climbmix": 0.5, "code": 0.5})
        sample_batch, state = next(gen)
        assert len(sample_batch) > 0
        assert isinstance(sample_batch[0], str)

        del gen
        gc.collect()


def test_terminal_source_streaming(monkeypatch):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_term:
        commands = [
            "ls -la /var/log | grep error",
            "chmod +x deploy.sh && ./deploy.sh",
            "git status -s",
            "ps aux | grep python",
        ]
        create_dummy_parquet(os.path.join(tmp_term, "shard_00000.parquet"), commands * 5, col_name="command")
        test_sources = {"terminal": {"dir": tmp_term}}
        monkeypatch.setattr("nanochat.dataset.PRETRAIN_SOURCES", test_sources)
        monkeypatch.setattr("nanochat.dataloader.PRETRAIN_SOURCES", test_sources)

        gen = _document_batches("train", None, tokenizer_batch_size=4, domain_weights={"terminal": 1.0})
        sample_batch, _ = next(gen)
        assert len(sample_batch) > 0
        assert any("chmod +x" in s or "git status" in s for s in sample_batch)

        del gen
        gc.collect()
