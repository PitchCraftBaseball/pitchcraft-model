import json
import torch
from pathlib import Path
from datetime import datetime


def export_model(model, out_dir: str = None, filename: str = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "model_shared" / "trained-parameters"
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"pitch_rnn_{datetime.now().strftime('%Y%m%d')}.pt"

    save_path = out_path / filename
    torch.save(model.state_dict(), save_path)
    print(f"Model saved {save_path}")
    return save_path


def export_vocabs(cat_vocabs: dict, y_vocab: dict, feature_spec: dict, out_dir: str = None, filename: str = None) -> Path:
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "model_shared" / "vocab"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"rnn_vocab_{datetime.now().strftime('%Y%m%d')}.json"

    payload = {
        "cat_vocabs": {
            col: {str(k): v for k, v in vocab.items()}   # JSON keys must be strings
            for col, vocab in cat_vocabs.items()
        },
        "y_vocab": {str(k): v for k, v in y_vocab.items()},
        "feature_spec": feature_spec,
        "exported_at": datetime.now().isoformat(),
    }

    save_path = out_path / filename
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Vocabs saved {save_path}")
    return save_path

def export_test_tensors(Xc_te, Xn_te, Y_te, Yh_te, Yv_te, out_dir=None, filename=None):
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "model_shared" / "test_data"
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"test_tensors_{datetime.now().strftime('%Y%m%d')}.pt"

    save_path = out_path / filename
    torch.save({
        "Xc":     Xc_te,
        "Xn":     Xn_te,
        "Y":      Y_te,
        "Y_horiz": Yh_te,
        "Y_vert":  Yv_te,
    }, save_path)
    print(f"Test tensors saved {save_path}")
    return save_path

def load_vocabs(vocab_path: str) -> tuple[dict, dict, dict]:
    """
    Load vocabs and feature spec exported by export_vocabs().

    Returns:
        cat_vocabs, y_vocab, feature_spec
    """
    with open(vocab_path, "r") as f:
        payload = json.load(f)

    # Restore original key types — pitcher/batter IDs are ints, pitch types are strings
    cat_vocabs = {}
    for col, vocab in payload["cat_vocabs"].items():
        if col in ("pitcher", "batter"):
            cat_vocabs[col] = {int(k): v for k, v in vocab.items()}
        else:
            cat_vocabs[col] = {k: v for k, v in vocab.items()}

    y_vocab = {k: v for k, v in payload["y_vocab"].items()}
    feature_spec = payload["feature_spec"]

    return cat_vocabs, y_vocab, feature_spec

def get_latest_file(directory: str, pattern: str) -> Path:
    """Return the most recently modified file matching a glob pattern."""
    files = sorted(Path(directory).glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")
    return files[-1]