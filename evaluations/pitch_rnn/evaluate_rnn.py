import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import json
import torch.nn as nn

from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from model_shared.rnn_definition import PitchRNN
from model_shared.feature_engineering.pitch_constants import PAD_ID
from pitch_rnn.sequence_builder import PitchSeqDS
from pitch_rnn.export_artifacts import *
from sklearn.calibration import calibration_curve
import torch.nn.functional as F

# ── Load artifacts ────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent  # repo root (one level above pitch_rnn / model_eval)

def load_model_and_vocabs(vocab_path: str, model_path: str, emb_dims, num_layers, hidden):
    """Reconstruct PitchRNN from saved weights and vocab files, returning model, vocabs, and feature spec."""
    cat_vocabs, y_vocab, feature_spec = load_vocabs(vocab_path)
    NUM_COLS = feature_spec["num_cols"]

    cat_vocab_sizes = {c: len(cat_vocabs[c]) + 1 for c in feature_spec["cat_cols"]}
    num_classes = len(y_vocab) + 1

    model = PitchRNN(
        cat_vocab_sizes=cat_vocab_sizes,
        num_features=len(NUM_COLS),
        emb_dims=emb_dims,
        num_classes=num_classes,
        num_layers=num_layers,
        hidden=hidden
    )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    id_to_pitch = {v: k for k, v in y_vocab.items()}

    return model, cat_vocabs, y_vocab, id_to_pitch, feature_spec, num_classes


def load_test_loader(tensors_path: str, batch_size: int = 64) -> DataLoader:
    """Load saved test tensors from disk and wrap them in a shuffled-off DataLoader."""
    tensors = torch.load(tensors_path, map_location="cpu")
    dataset = PitchSeqDS(tensors["Xc"], tensors["Xn"], tensors["Y"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_arsenals(arsenals_path: str = None) -> dict:
    """Load the pitcher arsenal JSON, defaulting to the shared arsenals_all.json in the repo."""
    if arsenals_path is None:
        arsenals_path = BASE.parent / "pitch_arsenal" / "arsenals_all.json"
    with open(arsenals_path) as f:
        arsenals = json.load(f)
    print("Got Arsenals")
    return arsenals

def build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year):
    """Return a (num_pitchers+1, num_classes) mask zeroing out pitches outside each pitcher's known arsenal."""
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

def get_all_predictions(model, test_loader, device, arsenal_masks=None, pad_id=PAD_ID, temperature=1.0):
    """Run inference over the full test loader and return flattened arrays of true and predicted labels."""
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y     = y.to(device)

            logits = model(x_cat, x_num)
            logits = logits / temperature 

            if arsenal_masks is not None:
                pitcher_ids   = x_cat[:, :, 0]
                pitch_mask    = arsenal_masks[pitcher_ids].to(device)
                logits        = logits.masked_fill(pitch_mask == 0, float("-inf"))

            preds = logits.argmax(dim=-1)
            valid = (y != pad_id).bool()

            all_preds.append(preds[valid].cpu().numpy())
            all_true.append(y[valid].cpu().numpy())

    return np.concatenate(all_true), np.concatenate(all_preds)


# ── Evaluation functions ──────────────────────────────────────────────────────

def get_accuracy(model, test_loader, device, arsenal_masks=None, pad_id=PAD_ID, temperature = 1.0):
    """Compute and print top-1 token accuracy over all non-pad positions."""
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id, temperature)
    accuracy = 100 * (y_pred == y_true).sum() / len(y_true)
    print(f"Token Accuracy (no PAD): {accuracy:.2f}%")
    return accuracy


def get_top_k_accuracy(model, test_loader, device, arsenal_masks=None, K=3, pad_id=PAD_ID, temperature = 1.0):
    """Compute and print top-K accuracy, checking whether the true label falls in the K highest-scoring predictions."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y     = y.to(device)

            logits = model(x_cat, x_num)
            logits = logits / temperature 

            if arsenal_masks is not None:
                pitcher_ids = x_cat[:, :, 0]
                pitch_mask  = arsenal_masks[pitcher_ids].to(device)
                logits      = logits.masked_fill(pitch_mask == 0, float("-inf"))

            topk  = logits.topk(K, dim=-1).indices
            valid = (y != pad_id)
            match = (topk == y.unsqueeze(-1)).any(dim=-1)
            correct += (match & valid).sum().item()
            total   += valid.sum().item()

    accuracy = 100 * correct / total
    print(f"Top-{K} Accuracy: {accuracy:.2f}%")
    return accuracy


def get_most_common_pitches(model, test_loader, device, id_to_pitch, arsenal_masks=None, top_n=5, pad_id=PAD_ID, temperature=1.0):
    """Tally and print the top_n most frequently predicted pitch types across the test set."""
    model.eval()
    counts = Counter()

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)

            logits = model(x_cat, x_num)
            logits = logits / temperature 

            if arsenal_masks is not None:
                pitcher_ids = x_cat[:, :, 0]
                pitch_mask  = arsenal_masks[pitcher_ids].to(device)
                logits      = logits.masked_fill(pitch_mask == 0, float("-inf"))

            preds = logits.argmax(dim=-1)
            for p in preds.view(-1).tolist():
                if p != pad_id:
                    counts[p] += 1

    print(f"Top {top_n} most predicted pitches:")
    for pid, cnt in counts.most_common(top_n):
        print(f"  {id_to_pitch.get(pid, 'PAD')}: {cnt}")

    return counts.most_common(top_n)


def print_classification_report(model, test_loader, device, id_to_pitch, arsenal_masks=None, pad_id=PAD_ID, temperature = 1.0):
    """Print and return a per-class precision/recall/F1 classification report."""
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id, temperature)
    labels       = sorted(id_to_pitch.keys())
    target_names = [id_to_pitch[l] for l in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names)
    print(report)
    return report

def generate_confusion_matrix(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, pad_id=PAD_ID,
    figsize=(12, 10), save_dir: str = None,
    temperature = 1.0
):
    """Plot a row-normalized confusion matrix heatmap and optionally save it to disk."""
    y_true, y_pred = get_all_predictions(model, test_loader, device, arsenal_masks, pad_id, temperature)

    labels      = sorted(id_to_pitch.keys())
    tick_labels = [id_to_pitch[l] for l in labels]

    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    n = len(labels)
    cell_inches = 0.55
    if figsize == (12, 10):
        figsize = (n * cell_inches + 3.5, n * cell_inches + 2.5)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cmap="Purples",
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": 16},
    )
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Predicted Pitch Type", fontsize=14)
    ax.set_ylabel("True Pitch Type", fontsize=14)
    ax.set_title("Pitch Type Confusion Matrix", fontsize=18)
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

def generate_calibration_curves(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, pad_id=PAD_ID,
    n_bins=5, figsize=(8, 7), save_dir: str = None
):
    """Plot per-class calibration curves comparing predicted probabilities to observed rates."""
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)

            logits = model(x_cat, x_num)

            if arsenal_masks is not None:
                pitcher_ids = x_cat[:, :, 0]
                pitch_mask  = arsenal_masks[pitcher_ids].to(device)
                logits      = logits.masked_fill(pitch_mask == 0, float("-inf"))

            all_logits.append(logits.reshape(-1, logits.size(-1)).cpu())
            all_targets.append(y.reshape(-1).cpu())

    all_logits  = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Remove PAD tokens
    mask        = all_targets != pad_id
    probs       = F.softmax(all_logits[mask], dim=-1).numpy()
    targets_np  = all_targets[mask].numpy()

    GROUP_COLORS = {
        "fastball": ("tab:red",   "Fastball"),
        "breaking": ("tab:blue",  "Breaking"),
        "offspeed": ("tab:green", "Offspeed"),
    }

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect", zorder=0)

    for class_id in sorted(id_to_pitch.keys()):
        pitch_name     = id_to_pitch[class_id]
        binary_targets = (targets_np == class_id).astype(int)
        class_probs    = probs[:, class_id]

        if binary_targets.sum() == 0:
            continue

        fraction_pos, mean_pred = calibration_curve(
            binary_targets, class_probs,
            n_bins=n_bins, strategy="uniform"
        )

        color, legend_label = GROUP_COLORS.get(pitch_name.lower(), ("tab:gray", pitch_name))
        ax.plot(mean_pred, fraction_pos, marker="o", color=color, label=legend_label, alpha=0.85)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability", fontsize=18)
    ax.set_ylabel("Fraction Positive", fontsize=18)
    ax.set_title("Calibration Curves by Pitch Type", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename  = f"calibration_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = out_path / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Calibration curves saved {save_path}")

    plt.show()

def find_optimal_temperature(model, test_loader, device, pad_id=PAD_ID):
    """Use LBFGS to find the temperature scalar that minimizes NLL on the test set."""
    model.eval()
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)

            logits = model(x_cat, x_num)
            all_logits.append(logits.reshape(-1, logits.size(-1)).cpu())
            all_targets.append(y.reshape(-1).cpu())

    all_logits  = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Remove PAD
    mask        = all_targets != pad_id
    all_logits  = all_logits[mask]
    all_targets = all_targets[mask]

    # Learnable temperature parameter
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion   = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = all_logits / temperature.clamp(min=0.1)
        loss = criterion(scaled_logits, all_targets)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    optimal_temp = temperature.item()
    return optimal_temp

def get_positional_accuracy(model, test_loader, device, arsenal_masks=None, pad_id=PAD_ID, temperature=1.0):
    """Report per-pitch-position accuracy within a plate appearance sequence."""
    model.eval()
    # Accumulate correct/total per sequence position
    from collections import defaultdict
    correct_by_pos = defaultdict(int)
    total_by_pos   = defaultdict(int)

    with torch.no_grad():
        for x_cat, x_num, y in test_loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y     = y.to(device)

            logits = model(x_cat, x_num) / temperature  # (batch, seq_len, num_classes)

            if arsenal_masks is not None:
                pitcher_ids = x_cat[:, :, 0]
                pitch_mask  = arsenal_masks[pitcher_ids].to(device)
                logits      = logits.masked_fill(pitch_mask == 0, float("-inf"))

            preds = logits.argmax(dim=-1)  # (batch, seq_len)
            valid = (y != pad_id)          # (batch, seq_len)

            seq_len = y.size(1)
            for pos in range(seq_len):
                mask_pos    = valid[:, pos]
                correct_pos = (preds[:, pos] == y[:, pos]) & mask_pos
                correct_by_pos[pos] += correct_pos.sum().item()
                total_by_pos[pos]   += mask_pos.sum().item()

    print("Positional Accuracy:")
    print(f"  {'Position':<12} {'Accuracy':>10} {'Sample N':>10}")
    print(f"  {'-'*34}")
    results = {}
    for pos in sorted(correct_by_pos.keys()):
        if total_by_pos[pos] > 0:
            acc = 100 * correct_by_pos[pos] / total_by_pos[pos]
            print(f"  Pitch {pos+1:<6} {acc:>9.2f}%  {total_by_pos[pos]:>9,}")
            results[pos + 1] = {"accuracy": acc, "n": total_by_pos[pos]}

    return results

# ── Master evaluation ─────────────────────────────────────────────────────────

def evaluate_model_complete(
    model, test_loader, device, id_to_pitch,
    arsenal_masks=None, pad_id=PAD_ID,
    confusion_matrix_save_dir: str = None,
    temperature: float = 1.0,
):
    """Run all evaluation functions and return a summary dict of metrics and plots."""
    print("\n1. Token Accuracy:")
    acc = get_accuracy(model, test_loader, device, arsenal_masks, pad_id, temperature=temperature)

    print("\n2. Top-3 Accuracy:")
    top3 = get_top_k_accuracy(model, test_loader, device, arsenal_masks, K=2, pad_id=pad_id, temperature=temperature)

    print("\n3. Most Common Predictions:")
    common = get_most_common_pitches(model, test_loader, device, id_to_pitch, arsenal_masks, pad_id=pad_id, temperature=temperature)

    print("\n4. Detailed Classification Report:")
    class_report = print_classification_report(model, test_loader, device, id_to_pitch, arsenal_masks, pad_id, temperature)

    print("\n5. Confusion Matrix:")
    cm, cm_norm = generate_confusion_matrix(
        model, test_loader, device, id_to_pitch,
        arsenal_masks=arsenal_masks,
        pad_id=pad_id,
        save_dir=confusion_matrix_save_dir,
        temperature=temperature
    )

    print("\n6. Calibration Curves:")
    generate_calibration_curves(
        model, test_loader, device, id_to_pitch,
        arsenal_masks=arsenal_masks,
        pad_id=pad_id,
        save_dir=confusion_matrix_save_dir,  # saves to same eval output folder
    )
    
    positional_results = get_positional_accuracy(
        model, test_loader, device,
        arsenal_masks=arsenal_masks,
        temperature=temperature
    )   

    return {
        "accuracy":                   acc,
        "top3_accuracy":              top3,
        "most_common":                common,
        "classification_report":      class_report,
        "confusion_matrix":           cm,
        "confusion_matrix_normalized": cm_norm,
    }

def evaluate_rnn(emb_dims, num_layers, use_arsenal_mask, hidden):
    """Top-level entry point: load the latest artifacts, calibrate temperature, and run full evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE = Path(__file__).parent.parent.parent  # evaluations/pitch_rnn -> evaluations -> repo root

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}"

    vocab_path = get_latest_file(BASE / "model_shared" / "vocab", "rnn_vocab_*.json")
    model_path = get_latest_file(BASE / "model_shared" / "trained-parameters", "pitch_rnn_*.pt")
    tensors_path = get_latest_file(BASE / "model_shared" / "test_data", "test_tensors_*.pt")

    model, cat_vocabs, y_vocab, id_to_pitch, feature_spec, num_classes = load_model_and_vocabs(
        vocab_path = vocab_path,
        model_path = model_path,
        emb_dims=emb_dims,
        num_layers=num_layers,
        hidden=hidden
    )
    model = model.to(device)

    test_loader  = load_test_loader(tensors_path)
    arsenals     = load_arsenals()

    if use_arsenal_mask:
        arsenal_masks = build_arsenal_masks(arsenals, cat_vocabs, y_vocab, num_classes, year="2025")
    else: 
        arsenal_masks = None

    optimal_temp = find_optimal_temperature(model, test_loader, device)

    temp_path = BASE / "model_shared" / "vocab" / "temperature.json"
    with open(temp_path, "w") as f:
        json.dump({"temperature": optimal_temp}, f)

    # ── Evaluate with temperature applied ────────────────────────────────────
    evaluate_model_complete(
        model, test_loader, device, id_to_pitch,
        arsenal_masks=arsenal_masks,
        temperature=optimal_temp,           # ADD THIS
        confusion_matrix_save_dir=f"evaluations/pitch_rnn/eval_output/{stem}",
    )