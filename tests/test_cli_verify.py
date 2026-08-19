"""Failing tests for `leann verify` (cross-artifact index integrity check)."""

import json
import pickle
from pathlib import Path

import numpy as np
from leann.cli import LeannCLI


def _write_faiss_index(path: Path, num_vectors: int, dim: int = 4) -> None:
    from leann_backend_hnsw import faiss

    index = faiss.IndexFlatL2(dim)
    vectors = np.ascontiguousarray(
        np.random.default_rng(0).random((num_vectors, dim), dtype=np.float32)
    )
    try:
        index.add(vectors)
    except TypeError:
        index.add(num_vectors, faiss.swig_ptr(vectors))
    faiss.write_index(index, str(path))


def _make_ivf_index(tmp_path: Path, passage_ids: list[str]) -> Path:
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    prefix = index_dir / "documents.leann"

    Path(str(prefix) + ".meta.json").write_text(
        json.dumps(
            {
                "backend_name": "ivf",
                "embedding_model": "m",
                "embedding_mode": "sentence-transformers",
                "dimensions": 4,
                "backend_kwargs": {},
            }
        ),
        encoding="utf-8",
    )

    offsets: dict[str, int] = {}
    with open(str(prefix) + ".passages.jsonl", "wb") as f:
        for pid in passage_ids:
            offsets[pid] = f.tell()
            line = json.dumps({"id": pid, "text": f"passage {pid}", "metadata": {}}) + "\n"
            f.write(line.encode("utf-8"))
    with open(str(prefix) + ".passages.idx", "wb") as f:
        pickle.dump(offsets, f)

    id_map = {
        "id_to_passage": {str(i): pid for i, pid in enumerate(passage_ids)},
        "passage_to_id": {pid: i for i, pid in enumerate(passage_ids)},
        "next_id": len(passage_ids),
    }
    Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json").write_text(
        json.dumps(id_map), encoding="utf-8"
    )

    _write_faiss_index(
        Path(str(prefix).removesuffix(".leann") + ".index"), num_vectors=len(passage_ids)
    )
    return prefix


def _run_verify(index_name: str = "idx") -> int:
    cli = LeannCLI()
    args = cli.create_parser().parse_args(["verify", index_name])
    return int(cli.verify_command(args) or 0)


def test_verify_passes_on_healthy_ivf_index(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    _make_ivf_index(tmp_path, ["0", "1", "2"])

    # Act
    rc = _run_verify()

    # Assert
    assert rc == 0


def test_verify_fails_on_truncated_passages_jsonl(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    jsonl = Path(str(prefix) + ".passages.jsonl")
    lines = jsonl.read_bytes().splitlines(keepends=True)
    jsonl.write_bytes(b"".join(lines[:-1]))

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0
    captured = capsys.readouterr()
    assert (captured.out + captured.err).strip()


def test_verify_fails_when_id_map_not_exact_inverse(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    id_map_path = Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json")
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["passage_to_id"]["2"] = 0
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_fails_on_bad_offset_in_passages_idx(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    idx_path = Path(str(prefix) + ".passages.idx")
    with open(idx_path, "rb") as f:
        offsets = pickle.load(f)
    offsets["1"] = offsets["1"] + 3
    with open(idx_path, "wb") as f:
        pickle.dump(offsets, f)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_fails_when_id_map_has_id_missing_from_passages(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    prefix = _make_ivf_index(tmp_path, ["0", "1", "2"])
    id_map_path = Path(str(prefix).removesuffix(".leann") + ".ivf_id_map.json")
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["id_to_passage"]["3"] = "orphan"
    id_map["passage_to_id"]["orphan"] = 3
    id_map["next_id"] = 4
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    _write_faiss_index(Path(str(prefix).removesuffix(".leann") + ".index"), num_vectors=4)

    # Act
    rc = _run_verify()

    # Assert
    assert rc != 0


def test_verify_reports_finding_instead_of_crashing_on_non_numeric_id_keys(
    tmp_path, monkeypatch, capsys
):
    # Arrange: every id_to_passage key is non-numeric (the exact corruption verify flags)
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    id_map_path = prefix.parent / "documents.ivf_id_map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    id_map["id_to_passage"] = {"abc": "p0", "xyz": "p1"}
    id_map["passage_to_id"] = {"p0": "abc", "p1": "xyz"}
    id_map_path.write_text(json.dumps(id_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert: findings printed, no ValueError traceback
    assert rc == 1
    assert "not an integer" in out


def test_verify_unreadable_idx_does_not_emit_misleading_cross_findings(
    tmp_path, monkeypatch, capsys
):
    # Arrange
    prefix = _make_ivf_index(tmp_path, ["p0", "p1"])
    Path(str(prefix) + ".passages.idx").write_bytes(b"not a pickle")
    monkeypatch.chdir(tmp_path)

    # Act
    rc = _run_verify()
    out = capsys.readouterr().out

    # Assert: the unreadable idx is the finding; no derived mismatch noise
    assert rc == 1
    assert "passages.idx unreadable" in out
    assert "do not match passages.idx" not in out
    assert "passages.jsonl has" not in out
