import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json

from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from model_shared.rnn_definition import PitchRNN
from model_shared.feature_engineering.pitch_constants import PAD_ID
from pitch_rnn.sequence_builder import PitchSeqDS
from pitch_rnn.export_artifacts import *

# ── Load artifacts ────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent  # repo root (one level above pitch_rnn / model_eval)

def load_model_and_vocabs(vocab_path, model_path, emb_dims, num_layers):
    cat_vocabs, y_vocab, feature_spec = load_vocabs(vocab_path)
    NUM_COLS = feature_spec["num_cols"]

    cat_vocab_sizes = {c: len(cat_vocabs[c]) + 1 for c in feature_spec["cat_cols"]}
    num_classes = len(y_vocab) + 1

    model = PitchRNN(
        cat_vocab_sizes      = cat_vocab_sizes,
        num_features         = len(NUM_COLS),
        emb_dims             = emb_dims,
        num_classes          = num_classes,
        num_location_classes = 3,
        num_layers           = num_layers,
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    id_to_pitch = {v: k for k, v in y_vocab.items()}
    return model, cat_vocabs, y_vocab, id_to_pitch, feature_spec, num_classes


def load_test_loader(tensors_path: str, batch_size: int = 64) -> DataLoader:
    tensors = torch.load(tensors_path, map_location="cpu")
    dataset = PitchSeqDS(
        tensors["Xc"],
        tensors["Xn"],
        tensors["Y"],
        tensors["Y_horiz"],
        tensors["Y_vert"],
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_arsenals(arsenals_path: str = None) -> dict:
    if arsenals_path is None:
        arsenals_path = BASE.parent / "pitch_arsenal" / "arsenals_all.json"
    with open(arsenals_path) as f:
        arsenals = json.load(f)
    print("Got Arsenals")
    return arsenals

def build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year):
    pitcher_vocab = cat_vocabs["pitcher"]
        
    masks = torch.ones(len(pitcher_vocab) + 1, num_classes)

    for pitcher_str, data in arsenals.items():
        pitcher_id_int = int(pitcher_str)

        if pitcher_id_int not in pitcher_vocab:
            continue

        enc_id = pitcher_vocab[pitcher_id_int]

        if year in data:
            allowed_pitches = data[year]["arsenal_mask"]
        elif str(int(year) - 1) in data:
            allowed_pitches = data[str(int(year) - 1)]["arsenal_mask"]
        else:
            continue

        masks[enc_id] = 0
        for pitch in allowed_pitches:
            if pitch in y_vocab:
                masks[enc_id, y_vocab[pitch]] = 1

    return masks

# ── Prediction helpers ────────────────────────────────────────────────────────

def get_all_predictions(
    model, test_loader, device,
    arsenal_masks=None, pad_id=PAD_ID
):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for x_cat, x_num, y, y_horiz, y_vert in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y     = y.to(device)

            logits_pitch, _, _ = model(x_cat, x_num)

            if arsenal_masks is not None:
                pitcher_ids = x_cat[:, :, 0]
                pitch_mask  = arsenal_masks[pitcher_ids].to(device)
                logits_pitch = logits_pitch.masked_fill(pitch_mask == 0, float("-inf"))

            preds = logits_pitch.argmax(dim=-1)
            valid = (y != pad_id).bool()

            all_preds.append(preds[valid].cpu().numpy())
            all_true.append(y[valid].cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_preds)


# ── Evaluation functions ──────────────────────────────────────────────────────

def get_accuracy(model, test_loader, device, arsenal_masks=None, pad_id=PAD_ID):
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id)
    accuracy = 100 * (y_pred == y_true).sum() / len(y_true)
    print(f"Token Accuracy (no PAD): {accuracy:.2f}%")
    return accuracy


def get_top_k_accuracy(model, test_loader, device,arsenal_masks=None, K=2, pad_id=PAD_ID):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x_cat, x_num, y, y_horiz, y_vert in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y     = y.to(device)

            logits_pitch, _, _ = model(x_cat, x_num)

            if arsenal_masks is not None:
                pitcher_ids  = x_cat[:, :, 0]
                pitch_mask   = arsenal_masks[pitcher_ids].to(device)
                logits_pitch = logits_pitch.masked_fill(pitch_mask == 0, float("-inf"))

            topk  = logits_pitch.topk(K, dim=-1).indices
            valid = (y != pad_id)
            match = (topk == y.unsqueeze(-1)).any(dim=-1)
            correct += (match & valid).sum().item()
            total   += valid.sum().item()

    accuracy = 100 * correct / total
    print(f"Top-{K} Accuracy: {accuracy:.2f}%")
    return accuracy


def get_most_common_pitches(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, top_n=5, pad_id=PAD_ID
):
    model.eval()
    counts = Counter()

    with torch.no_grad():
        for x_cat, x_num, y, y_horiz, y_vert in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)

            logits_pitch, _, _ = model(x_cat, x_num)

            if arsenal_masks is not None:
                pitcher_ids  = x_cat[:, :, 0]
                pitch_mask   = arsenal_masks[pitcher_ids].to(device)
                logits_pitch = logits_pitch.masked_fill(pitch_mask == 0, float("-inf"))

            preds = logits_pitch.argmax(dim=-1)
            for p in preds.view(-1).tolist():
                if p != pad_id:
                    counts[p] += 1

    print(f"Top {top_n} most predicted pitches:")
    for pid, cnt in counts.most_common(top_n):
        print(f"  {id_to_pitch.get(pid, 'PAD')}: {cnt}")

    return counts.most_common(top_n)

def get_location_accuracy(
    model, test_loader, device, pad_id=PAD_ID
):
    """
    Evaluates horiz and vert head accuracy independently.
    Masks out PAD positions where location target was missing.
    """
    model.eval()
    h_correct, h_total = 0, 0
    v_correct, v_total = 0, 0

    with torch.no_grad():
        for x_cat, x_num, y, y_horiz, y_vert in test_loader:
            x_cat   = x_cat.to(device)
            x_num   = x_num.to(device)
            y_horiz = y_horiz.to(device)
            y_vert  = y_vert.to(device)

            _, logits_horiz, logits_vert = model(x_cat, x_num)

            pred_h = logits_horiz.argmax(dim=-1)
            pred_v = logits_vert.argmax(dim=-1)

            mask_h = (y_horiz != pad_id)
            mask_v = (y_vert  != pad_id)

            h_correct += ((pred_h == y_horiz) & mask_h).sum().item()
            h_total   += mask_h.sum().item()
            v_correct += ((pred_v == y_vert)  & mask_v).sum().item()
            v_total   += mask_v.sum().item()

    horiz_acc = 100 * h_correct / max(1, h_total)
    vert_acc  = 100 * v_correct / max(1, v_total)

    print(f"Horizontal Location Accuracy: {horiz_acc:.2f}%")
    print(f"Vertical Location Accuracy:   {vert_acc:.2f}%")

    return horiz_acc, vert_acc


def get_joint_accuracy(
    model, test_loader, device, pad_id=PAD_ID
):
    """
    Tracks how often pitch type, horiz, and vert are
    all correct simultaneously.
    """
    model.eval()
    joint_correct, total = 0, 0

    with torch.no_grad():
        for x_cat, x_num, y, y_horiz, y_vert in test_loader:
            x_cat   = x_cat.to(device)
            x_num   = x_num.to(device)
            y       = y.to(device)
            y_horiz = y_horiz.to(device)
            y_vert  = y_vert.to(device)

            logits_pitch, logits_horiz, logits_vert = model(x_cat, x_num)

            pred_p = logits_pitch.argmax(dim=-1)
            pred_h = logits_horiz.argmax(dim=-1)
            pred_v = logits_vert.argmax(dim=-1)

            # only count positions where all three targets are valid
            mask = (
                (y       != pad_id) &
                (y_horiz != pad_id) &
                (y_vert  != pad_id)
            )

            all_correct = (
                (pred_p == y)       &
                (pred_h == y_horiz) &
                (pred_v == y_vert)  &
                mask
            )

            joint_correct += all_correct.sum().item()
            total         += mask.sum().item()

    joint_acc = 100 * joint_correct / max(1, total)
    print(f"Joint Accuracy (pitch + horiz + vert): {joint_acc:.2f}%")
    return joint_acc


def print_classification_report(model, test_loader, device, id_to_pitch, arsenal_masks=None, pad_id=PAD_ID):
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id)
    labels       = sorted(id_to_pitch.keys())
    target_names = [id_to_pitch[l] for l in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names)
    print(report)
    return report

def generate_confusion_matrix(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, pad_id=PAD_ID,
    figsize=(12, 10), save_dir: str = None
):
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id)

    labels      = sorted(id_to_pitch.keys())
    tick_labels = [id_to_pitch[l] for l in labels]

    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        ax=ax,
    )
    ax.set_xlabel("Predicted Pitch Type")
    ax.set_ylabel("True Pitch Type")
    ax.set_title("Pitch Type Confusion Matrix (Token-Level)")
    plt.tight_layout()

    if save_dir is not None:
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename  = f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = out_path / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved {save_path}")

    plt.show()
    return cm, cm_norm


# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_model_complete(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, pad_id=PAD_ID,
    confusion_matrix_save_dir=None,
):
    print("\n1. Token Accuracy:")
    acc = get_accuracy(model, test_loader, device, arsenal_masks, pad_id)

    print("\n2. Top-3 Accuracy:")
    top3 = get_top_k_accuracy(model, test_loader, device, arsenal_masks, K=3, pad_id=pad_id)

    print("\n3. Most Common Predictions:")
    common = get_most_common_pitches(model, test_loader, device, id_to_pitch, arsenal_masks, pad_id=pad_id)

    print("\n4. Detailed Classification Report:")
    class_report = print_classification_report(model, test_loader, device, id_to_pitch, arsenal_masks, pad_id)

    print("\n5. Location Accuracy:")
    horiz_acc, vert_acc = get_location_accuracy(model, test_loader, device, pad_id)

    print("\n6. Joint Accuracy (pitch + location):")
    joint_acc = get_joint_accuracy(model, test_loader, device, pad_id)

    print("\n7. Confusion Matrix:")
    cm, cm_norm = generate_confusion_matrix(
        model, test_loader, device, id_to_pitch,
        arsenal_masks=arsenal_masks,
        pad_id=pad_id,
        save_dir=confusion_matrix_save_dir,
    )

    results = {
        "accuracy":                    acc,
        "top3_accuracy":               top3,
        "horiz_accuracy":              horiz_acc,
        "vert_accuracy":               vert_acc,
        "joint_accuracy":              joint_acc,
        "most_common":                 common,
        "classification_report":       class_report,
        "confusion_matrix":            cm,
        "confusion_matrix_normalized": cm_norm,
    }

    if confusion_matrix_save_dir is not None:
        save_evaluation_results(
            results=results,
            save_dir=confusion_matrix_save_dir,
            run_label=None,
        )

    return results

def save_evaluation_results(results: dict, save_dir: str, run_label: str = None): 
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = run_label or "eval"
    timestamp = datetime.now().strftime("%Y%m%d")
    stem = f"{label}_{timestamp}"

    metrics = {
        "run_label":        label,
        "timestamp":        timestamp,
        "accuracy":         float(results["accuracy"]),
        "top3_accuracy":    float(results["top3_accuracy"]),
        "horiz_accuracy":   float(results["horiz_accuracy"]),
        "vert_accuracy":    float(results["vert_accuracy"]),
        "joint_accuracy":   float(results["joint_accuracy"]),
        "most_common": [
            {"class_id": int(cid), "count": int(cnt)}
            for cid, cnt in results["most_common"]
        ],
        "classification_report": results["classification_report"],
    }

    metrics_path = out_dir / f"{stem}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved {metrics_path}")

     # ── 2. Confusion matrices → .npy ─────────────────────────────────────────
    cm      = results.get("confusion_matrix")
    cm_norm = results.get("confusion_matrix_normalized")
 
    if cm is not None:
        cm_path = out_dir / f"{stem}_confusion_matrix.npy"
        np.save(cm_path, cm)
        print(f"CM (raw) saved {cm_path}")
 
    if cm_norm is not None:
        cm_norm_path = out_dir / f"{stem}_confusion_matrix_norm.npy"
        np.save(cm_norm_path, cm_norm)
        print(f"CM (norm) saved {cm_norm_path}")
 
    return out_dir


def evaluate_rnn(emb_dims, num_layers, use_arsenal_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE = Path(__file__).parent.parent.parent  # evaluations/pitch_rnn -> evaluations -> repo root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}"

    vocab_path = get_latest_file(BASE / "model_shared" / "vocab", "rnn_vocab_*.json")
    model_path = get_latest_file(BASE / "model_shared" / "trained-parameters", "pitch_rnn_*.pt")
    tensors_path = get_latest_file(BASE / "model_shared" / "test-data", "test_tensors_*.pt")

    model, cat_vocabs, y_vocab, id_to_pitch, feature_spec, num_classes = load_model_and_vocabs(
        vocab_path = vocab_path,
        model_path = model_path,
        emb_dims=emb_dims,
        num_layers=num_layers
    )
    model = model.to(device)

    test_loader  = load_test_loader(tensors_path)
    arsenals     = load_arsenals()

    if use_arsenal_mask:
        arsenal_masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year="2025")
    else: 
        arsenal_masks = None

    evaluate_model_complete(
        model, test_loader, device, id_to_pitch,
        arsenal_masks=arsenal_masks,
        confusion_matrix_save_dir=f"evaluations/pitch_rnn/eval_output/{stem}",
    )