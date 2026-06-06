import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pitch_rnn.export_artifacts as ea_mod
from pitch_rnn.export_artifacts import (
    export_model,
    export_test_tensors,
    export_vocabs,
    get_latest_file,
    load_vocabs,
)



def _make_model():
    m = MagicMock()
    m.state_dict.return_value = {"w": torch.zeros(2)}
    return m


def _make_tensors():
    return (
        torch.zeros(4, 3, dtype=torch.long),
        torch.zeros(4, 3, 2),
        torch.ones(4, 3, dtype=torch.long),
    )


def _write_vocab(tmp_path, cat_vocabs, y_vocab, feature_spec, filename="v.json"):
    return export_vocabs(
        cat_vocabs, y_vocab, feature_spec, out_dir=str(tmp_path), filename=filename
    )


def test_export_model_custom_dir_and_filename(tmp_path):
    """out_dir and filename both provided — file lands at the exact path."""
    result = export_model(_make_model(), out_dir=str(tmp_path), filename="m.pt")
    assert result == tmp_path / "m.pt"
    assert result.exists()


def test_export_model_default_filename_is_datestamped(tmp_path):
    """filename=None → pitch_rnn_YYYYMMDD.pt."""
    result = export_model(_make_model(), out_dir=str(tmp_path))
    assert result.name.startswith("pitch_rnn_")
    assert result.suffix == ".pt"


def test_export_model_default_dir(tmp_path, monkeypatch):
    """out_dir=None → file is placed in model_shared/trained-parameters relative to the module."""
    monkeypatch.setattr(
        ea_mod,
        "__file__",
        str(tmp_path / "pitch_rnn" / "export_artifacts.py"),
    )
    result = export_model(_make_model(), filename="test.pt")
    assert result == tmp_path / "model_shared" / "trained-parameters" / "test.pt"
    assert result.exists()


def test_export_model_creates_nested_directory(tmp_path):
    """mkdir(parents=True) is called — deep paths are created on demand."""
    result = export_model(_make_model(), out_dir=str(tmp_path / "a" / "b"), filename="m.pt")
    assert result.exists()


def test_export_model_state_dict_is_loadable(tmp_path):
    """torch.save uses model.state_dict(); the saved file is a valid checkpoint."""
    result = export_model(_make_model(), out_dir=str(tmp_path), filename="m.pt")
    loaded = torch.load(result)
    assert "w" in loaded


def test_export_vocabs_missing_cat_col_raises(tmp_path):
    """A cat_col declared in feature_spec but absent from cat_vocabs raises ValueError."""
    with pytest.raises(ValueError, match="pitcher"):
        export_vocabs(
            {}, {"FF": 1}, {"cat_cols": ["pitcher"]}, out_dir=str(tmp_path)
        )


def test_export_vocabs_multiple_missing_cols_all_listed(tmp_path):
    """All missing columns appear in the ValueError message."""
    with pytest.raises(ValueError) as exc:
        export_vocabs(
            {}, {"FF": 1}, {"cat_cols": ["pitcher", "stand"]}, out_dir=str(tmp_path)
        )
    assert "pitcher" in str(exc.value)
    assert "stand" in str(exc.value)


def test_export_vocabs_custom_dir_and_filename(tmp_path):
    """Happy path: file is created at the exact location specified."""
    result = _write_vocab(tmp_path, {"pitcher": {100: 1}}, {"FF": 1}, {"cat_cols": ["pitcher"]})
    assert result == tmp_path / "v.json"
    assert result.exists()


def test_export_vocabs_default_filename_is_datestamped(tmp_path):
    """filename=None → rnn_vocab_YYYYMMDD.json."""
    result = export_vocabs(
        {"pitcher": {100: 1}}, {"FF": 1}, {"cat_cols": ["pitcher"]}, out_dir=str(tmp_path)
    )
    assert result.name.startswith("rnn_vocab_")
    assert result.suffix == ".json"


def test_export_vocabs_default_dir(tmp_path, monkeypatch):
    """out_dir=None → file is placed in model_shared/vocab relative to the module."""
    monkeypatch.setattr(
        ea_mod,
        "__file__",
        str(tmp_path / "pitch_rnn" / "export_artifacts.py"),
    )
    result = export_vocabs(
        {"pitcher": {100: 1}}, {"FF": 1}, {"cat_cols": ["pitcher"]}, filename="v.json"
    )
    assert result == tmp_path / "model_shared" / "vocab" / "v.json"
    assert result.exists()


def test_export_vocabs_payload_has_required_keys(tmp_path):
    result = _write_vocab(tmp_path, {"pitcher": {100: 1}}, {"FF": 1, "SL": 2}, {"cat_cols": ["pitcher"]})
    payload = json.loads(result.read_text())
    assert {"cat_vocabs", "y_vocab", "feature_spec", "exported_at"} <= set(payload)


def test_export_vocabs_integer_keys_stringified(tmp_path):
    """int keys are written as strings in JSON (JSON spec requirement)."""
    result = _write_vocab(tmp_path, {"pitcher": {200: 1}}, {"FF": 1}, {"cat_cols": ["pitcher"]})
    payload = json.loads(result.read_text())
    assert "200" in payload["cat_vocabs"]["pitcher"]


def test_export_test_tensors_custom_dir_and_filename(tmp_path):
    Xc, Xn, Y = _make_tensors()
    result = export_test_tensors(Xc, Xn, Y, out_dir=str(tmp_path), filename="t.pt")
    assert result == tmp_path / "t.pt"
    assert result.exists()


def test_export_test_tensors_default_filename_is_datestamped(tmp_path):
    """filename=None → test_tensors_YYYYMMDD.pt."""
    Xc, Xn, Y = _make_tensors()
    result = export_test_tensors(Xc, Xn, Y, out_dir=str(tmp_path))
    assert result.name.startswith("test_tensors_")
    assert result.suffix == ".pt"


def test_export_test_tensors_default_dir(tmp_path, monkeypatch):
    """out_dir=None → file is placed in model_shared/test_data relative to the module."""
    monkeypatch.setattr(
        ea_mod,
        "__file__",
        str(tmp_path / "pitch_rnn" / "export_artifacts.py"),
    )
    Xc, Xn, Y = _make_tensors()
    result = export_test_tensors(Xc, Xn, Y, filename="t.pt")
    assert result == tmp_path / "model_shared" / "test_data" / "t.pt"
    assert result.exists()


def test_export_test_tensors_all_keys_present(tmp_path):
    """Saved file contains Xc, Xn, and Y keys."""
    Xc, Xn, Y = _make_tensors()
    result = export_test_tensors(Xc, Xn, Y, out_dir=str(tmp_path), filename="t.pt")
    loaded = torch.load(result)
    assert set(loaded.keys()) == {"Xc", "Xn", "Y"}


def test_load_vocabs_pitcher_keys_restored_as_ints(tmp_path):
    """pitcher vocab keys are int in export → should be int after load."""
    path = _write_vocab(
        tmp_path,
        {"pitcher": {100: 1, 200: 2}},
        {"FF": 1},
        {"cat_cols": ["pitcher"]},
    )
    cat_vocabs, _, _ = load_vocabs(str(path))
    assert all(isinstance(k, int) for k in cat_vocabs["pitcher"])


def test_load_vocabs_batter_keys_restored_as_ints(tmp_path):
    """batter is also an ID column — keys must be int after load."""
    path = _write_vocab(
        tmp_path,
        {"pitcher": {1: 0}, "batter": {42: 1}},
        {"FF": 1},
        {"cat_cols": ["pitcher", "batter"]},
    )
    cat_vocabs, _, _ = load_vocabs(str(path))
    assert all(isinstance(k, int) for k in cat_vocabs["batter"])


def test_load_vocabs_non_id_keys_remain_strings(tmp_path):
    """Columns other than pitcher/batter keep string keys."""
    path = _write_vocab(
        tmp_path,
        {"pitcher": {1: 0}, "stand": {"R": 1, "L": 2}},
        {"FF": 1},
        {"cat_cols": ["pitcher", "stand"]},
    )
    cat_vocabs, _, _ = load_vocabs(str(path))
    assert all(isinstance(k, str) for k in cat_vocabs["stand"])


def test_load_vocabs_y_vocab_returned_correctly(tmp_path):
    path = _write_vocab(tmp_path, {"pitcher": {1: 0}}, {"FF": 1, "SL": 2}, {"cat_cols": ["pitcher"]})
    _, y_vocab, _ = load_vocabs(str(path))
    assert y_vocab == {"FF": 1, "SL": 2}


def test_load_vocabs_feature_spec_round_trips(tmp_path):
    spec = {"cat_cols": ["pitcher"], "num_cols": ["inning"]}
    path = _write_vocab(tmp_path, {"pitcher": {1: 0}}, {"FF": 1}, spec)
    _, _, loaded_spec = load_vocabs(str(path))
    assert loaded_spec == spec


def test_get_latest_file_returns_most_recently_modified(tmp_path):
    older = tmp_path / "a.pt"
    newer = tmp_path / "b.pt"
    older.write_bytes(b"")
    newer.write_bytes(b"")

    now = time.time()
    os.utime(older, (now - 1, now - 1))
    os.utime(newer, (now, now))

    result = get_latest_file(str(tmp_path), "*.pt")
    assert result.name == "b.pt"


def test_get_latest_file_raises_when_no_match(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files matching"):
        get_latest_file(str(tmp_path), "*.pt")
