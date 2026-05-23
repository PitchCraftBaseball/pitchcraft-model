"""
interpretability_analysis.py

Three interpretability tools for the PitchRNN model:
  1. Permutation Importance  — which features matter most globally
  2. Embedding Analysis      — what the model learned about categorical values
  3. Hidden State Analysis   — how the model's internal state evolves within an AB

Usage (standalone):
    python interpretability_analysis.py

Or import individual functions into a notebook for interactive use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import torch
from datetime import datetime
from sklearn.decomposition import PCA

from model_shared.feature_engineering.pitch_constants import PAD_ID
from evaluations.pitch_rnn.evaluate_rnn import load_model_and_vocabs, load_test_loader
from pitch_rnn.export_artifacts import get_latest_file


# ──────────────────────────────────────────────────────────────────────────────
# 1. PERMUTATION IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

def _baseline_accuracy(model, test_loader, device, pad_id=PAD_ID):
    """Compute token accuracy on unmodified test data."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_cat, x_num, y, *_ in test_loader:
            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            logits, _, _ = model(x_cat, x_num)
            preds = logits.argmax(dim=-1)
            mask  = y != pad_id
            correct += ((preds == y) & mask).sum().item()
            total   += mask.sum().item()
    return 100 * correct / total


def _accuracy_with_permuted_cat(model, test_loader, device, feat_idx, pad_id=PAD_ID):
    """Accuracy when one categorical feature column is randomly shuffled."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_cat, x_num, y, *_ in test_loader:
            x_cat = x_cat.clone()
            # Shuffle across the batch dimension so sequence structure is kept
            # but the relationship between this feature and the target is broken
            perm = torch.randperm(x_cat.size(0))
            x_cat[:, :, feat_idx] = x_cat[perm, :, feat_idx]

            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            logits, _, _ = model(x_cat, x_num)
            preds = logits.argmax(dim=-1)
            mask  = y != pad_id
            correct += ((preds == y) & mask).sum().item()
            total   += mask.sum().item()
    return 100 * correct / total


def _accuracy_with_permuted_num(model, test_loader, device, feat_idx, pad_id=PAD_ID):
    """Accuracy when one numerical feature column is randomly shuffled."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_cat, x_num, y, *_ in test_loader:
            x_num = x_num.clone()
            perm  = torch.randperm(x_num.size(0))
            x_num[:, :, feat_idx] = x_num[perm, :, feat_idx]

            x_cat, x_num, y = x_cat.to(device), x_num.to(device), y.to(device)
            logits, _, _ = model(x_cat, x_num)
            preds = logits.argmax(dim=-1)
            mask  = y != pad_id
            correct += ((preds == y) & mask).sum().item()
            total   += mask.sum().item()
    return 100 * correct / total


def permutation_importance(
    model, test_loader, device, feature_spec,
    n_repeats=3, pad_id=PAD_ID, save_dir=None
):
    """
    Computes permutation importance for every cat and num feature.

    For each feature, shuffles its values across the batch n_repeats times
    and averages the accuracy drop vs. baseline. A larger drop = more important.

    Returns
    -------
    dict : {feature_name: mean_accuracy_drop}  sorted descending
    """
    cat_cols = feature_spec["cat_cols"]
    num_cols = feature_spec["num_cols"]

    print("Computing baseline accuracy...")
    baseline = _baseline_accuracy(model, test_loader, device, pad_id)
    print(f"  Baseline accuracy: {baseline:.2f}%\n")

    results = {}

    print("Permuting categorical features...")
    for idx, feat in enumerate(cat_cols):
        drops = []
        for r in range(n_repeats):
            acc = _accuracy_with_permuted_cat(model, test_loader, device, idx, pad_id)
            drops.append(baseline - acc)
            print(f"  {feat:25s}  repeat {r+1}/{n_repeats}  drop={baseline - acc:+.3f}%")
        results[feat] = float(np.mean(drops))

    print("\nPermuting numerical features...")
    for idx, feat in enumerate(num_cols):
        drops = []
        for r in range(n_repeats):
            acc = _accuracy_with_permuted_num(model, test_loader, device, idx, pad_id)
            drops.append(baseline - acc)
            print(f"  {feat:25s}  repeat {r+1}/{n_repeats}  drop={baseline - acc:+.3f}%")
        results[feat] = float(np.mean(drops))

    # Sort descending by importance
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    _plot_permutation_importance(results, baseline, save_dir)
    return results


def _plot_permutation_importance(results, baseline, save_dir=None):
    features = list(results.keys())
    drops    = list(results.values())

    # Color bars: positive drop = model relied on this feature (red = important)
    #             negative drop = shuffling actually helped (blue)
    colors = ["#c0392b" if d > 0 else "#2980b9" for d in drops]

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.5)))
    bars = ax.barh(features[::-1], drops[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean Accuracy Drop (%) — higher = more important")
    ax.set_title(f"Permutation Feature Importance\n(baseline accuracy: {baseline:.2f}%)")
    ax.set_xlim(min(drops) - 0.5, max(drops) + 0.5)

    for bar, val in zip(bars, drops[::-1]):
        ax.text(
            val + (0.05 if val >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%",
            va="center", ha="left" if val >= 0 else "right",
            fontsize=8
        )

    plt.tight_layout()

    if save_dir is not None:
        _save_fig(fig, save_dir, "permutation_importance")

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def analyze_embeddings(
    model, cat_vocabs, feature_name,
    top_n=None, save_dir=None
):
    """
    Extracts and visualises the learned embeddings for one categorical feature
    using PCA to project down to 2D.

    Parameters
    ----------
    model        : trained PitchRNN
    cat_vocabs   : dict of {feature_name: {value: encoded_id}}
    feature_name : which categorical feature to analyse
                   e.g. "count_state", "prev_pitch_type", "pitcher", "batter"
    top_n        : if set, only plot the top_n most common values
                   (useful for pitcher / batter which have huge vocabs)
    save_dir     : optional path to save the plot

    Returns
    -------
    coords : np.ndarray  (n_values, 2)   PCA coordinates
    labels : list[str]                   label for each point
    """
    if feature_name not in model.embs:
        raise ValueError(
            f"'{feature_name}' not in model embeddings. "
            f"Available: {list(model.embs.keys())}"
        )

    # Pull weight matrix — shape (vocab_size, emb_dim), row 0 is the PAD token
    weights   = model.embs[feature_name].weight.detach().cpu().numpy()
    id_to_val = {v: k for k, v in cat_vocabs[feature_name].items()}

    # Skip PAD row (index 0) and only keep rows that have a label
    valid_ids = [i for i in range(1, len(weights)) if i in id_to_val]

    if top_n is not None and len(valid_ids) > top_n:
        # Keep only the first top_n entries (order reflects training frequency
        # since build_vocab enumerates in order of first appearance)
        valid_ids = valid_ids[:top_n]

    emb_matrix = weights[valid_ids]                          # (n, emb_dim)
    labels     = [str(id_to_val[i]) for i in valid_ids]

    # PCA → 2D
    n_components = min(2, emb_matrix.shape[0], emb_matrix.shape[1])
    pca    = PCA(n_components=n_components)
    coords = pca.fit_transform(emb_matrix)                   # (n, 2)
    var    = pca.explained_variance_ratio_

    print(f"\n[{feature_name}] Embedding shape: {emb_matrix.shape}")
    print(f"  PCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")

    _plot_embeddings(coords, labels, feature_name, var, save_dir)
    return coords, labels


def analyze_all_small_embeddings(model, cat_vocabs, feature_spec, save_dir=None):
    """
    Convenience wrapper — runs embedding analysis on every categorical feature
    that has a small enough vocab to read on a single chart (<=50 unique values).
    For pitcher / batter it applies top_n=40 automatically.
    """
    large_vocab_features = {"pitcher", "batter"}
    results = {}

    for feat in feature_spec["cat_cols"]:
        vocab_size = len(cat_vocabs[feat])
        top_n = 40 if feat in large_vocab_features else None
        print(f"\n{'='*60}")
        print(f"Embedding analysis: {feat}  (vocab size: {vocab_size})")
        coords, labels = analyze_embeddings(
            model, cat_vocabs, feat, top_n=top_n, save_dir=save_dir
        )
        results[feat] = {"coords": coords, "labels": labels}

    return results


def _plot_embeddings(coords, labels, feature_name, var, save_dir=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colour points by index so similar vocab positions get similar hues
    colors = cm.tab20(np.linspace(0, 1, len(labels)))

    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, zorder=3)

    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            alpha=0.85,
        )

    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title(f"Learned Embeddings — {feature_name}\n(PCA projection)")
    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_dir is not None:
        _save_fig(fig, save_dir, f"embeddings_{feature_name}")

    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3. HIDDEN STATE ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _forward_with_hidden(model, x_cat, x_num):
    """
    Runs a forward pass and returns both the logits and the full hidden state
    sequence. Works with the existing PitchRNN without modifying its definition
    by hooking directly into the RNN submodule.
    """
    model.eval()
    hidden_states = {}

    def hook_fn(module, input, output):
        # output is (h_n_seq, h_n_last) for nn.RNN
        hidden_states["h"] = output[0].detach().cpu()   # (batch, seq_len, hidden)

    handle = model.rnn.register_forward_hook(hook_fn)

    with torch.no_grad():
        logits, _, _ = model(x_cat, x_num)

    handle.remove()
    return logits, hidden_states["h"]


def analyze_hidden_states_single_ab(
    model, x_cat_ab, x_num_ab, device,
    pitch_labels=None, save_dir=None
):
    """
    Analyses hidden state evolution for a single at-bat.

    Parameters
    ----------
    model       : trained PitchRNN
    x_cat_ab    : (seq_len, n_cat_features)  tensor — one AB, no batch dim
    x_num_ab    : (seq_len, n_num_features)  tensor — one AB, no batch dim
    pitch_labels: list of str — optional human-readable pitch labels per step
                  e.g. ["FF", "SL", "FF"] — must match the real pitches
    save_dir    : optional path to save plots

    Returns
    -------
    hidden : np.ndarray (seq_len, hidden_size)
    deltas : np.ndarray (seq_len-1,)  — L2 distance between consecutive states
    """
    x_cat_ab = x_cat_ab.unsqueeze(0).to(device)   # add batch dim → (1, L, C)
    x_num_ab = x_num_ab.unsqueeze(0).to(device)

    _, h = _forward_with_hidden(model, x_cat_ab, x_num_ab)
    hidden = h.squeeze(0).numpy()                  # (seq_len, hidden_size)

    # How much does the hidden state change pitch to pitch?
    deltas = np.linalg.norm(np.diff(hidden, axis=0), axis=1)

    # PCA trajectory in 2D
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(hidden)             # (seq_len, 2)
    var    = pca.explained_variance_ratio_

    _plot_hidden_state_trajectory(coords, deltas, var, pitch_labels, save_dir)
    _plot_hidden_state_heatmap(hidden, pitch_labels, save_dir)

    return hidden, deltas


def analyze_hidden_states_batch(
    model, test_loader, device,
    n_abs=8, id_to_pitch=None, pad_id=PAD_ID, save_dir=None
):
    """
    Runs hidden state analysis on the first n_abs at-bats in the test loader,
    then plots the average delta profile and an overlay of PCA trajectories.

    Parameters
    ----------
    n_abs       : how many at-bats to sample
    id_to_pitch : {encoded_id: pitch_str} — used to label each step
    """
    model.eval()
    all_hidden  = []   # list of (seq_len, hidden) arrays  (only real steps)
    all_deltas  = []   # list of delta vectors
    all_labels  = []   # list of label lists per AB

    for x_cat, x_num, y, *_ in test_loader:
        if len(all_hidden) >= n_abs:
            break

        _, h = _forward_with_hidden(
            model,
            x_cat.to(device),
            x_num.to(device),
        )
        # h : (batch, seq_len, hidden)
        # y : (batch, seq_len)

        for b in range(x_cat.size(0)):
            if len(all_hidden) >= n_abs:
                break

            real_mask  = (y[b] != pad_id).numpy()
            real_steps = int(real_mask.sum())
            if real_steps < 2:
                continue

            h_ab   = h[b, :real_steps].numpy()          # (real_steps, hidden)
            deltas = np.linalg.norm(np.diff(h_ab, axis=0), axis=1)

            labels = None
            if id_to_pitch is not None:
                labels = [
                    id_to_pitch.get(int(y[b, t].item()), "?")
                    for t in range(real_steps)
                ]

            all_hidden.append(h_ab)
            all_deltas.append(deltas)
            all_labels.append(labels)

    _plot_average_delta_profile(all_deltas, save_dir)
    _plot_pca_trajectory_overlay(all_hidden, all_labels, save_dir)

    print(f"\nAnalysed {len(all_hidden)} at-bats.")
    return all_hidden, all_deltas


# ── Hidden state plot helpers ─────────────────────────────────────────────────

def _plot_hidden_state_trajectory(coords, deltas, var, pitch_labels, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PCA trajectory
    ax = axes[0]
    n  = len(coords)
    colors_traj = cm.plasma(np.linspace(0.1, 0.9, n))

    for i in range(n - 1):
        ax.annotate(
            "", xy=coords[i + 1], xytext=coords[i],
            arrowprops=dict(arrowstyle="->", color=colors_traj[i], lw=1.5)
        )

    ax.scatter(coords[:, 0], coords[:, 1], c=colors_traj, s=80, zorder=5)

    for i, (x, y) in enumerate(coords):
        label = pitch_labels[i] if pitch_labels else f"p{i+1}"
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("Hidden State Trajectory (PCA)\nearly → late (dark → bright)")
    ax.grid(True, alpha=0.2)

    # Right: delta bar chart
    ax2 = axes[1]
    transitions = [
        f"{pitch_labels[i] if pitch_labels else f'p{i+1}'}"
        f"→{pitch_labels[i+1] if pitch_labels else f'p{i+2}'}"
        for i in range(len(deltas))
    ]
    bar_colors = cm.RdYlGn_r(deltas / deltas.max())
    ax2.bar(range(len(deltas)), deltas, color=bar_colors, edgecolor="white")
    ax2.set_xticks(range(len(deltas)))
    ax2.set_xticklabels(transitions, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("L2 Distance Between Hidden States")
    ax2.set_title("Hidden State Change per Pitch\n(larger = bigger context shift)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_dir is not None:
        _save_fig(fig, save_dir, "hidden_state_trajectory")
    plt.show()


def _plot_hidden_state_heatmap(hidden, pitch_labels, save_dir):
    """
    Heatmap of the hidden state matrix — rows = pitch steps, cols = hidden units.
    Makes it easy to see which hidden units light up at which points.
    """
    fig, ax = plt.subplots(figsize=(14, max(3, hidden.shape[0] * 0.6)))
    sns.heatmap(
        hidden,
        cmap="RdBu_r",
        center=0,
        yticklabels=pitch_labels if pitch_labels else [f"pitch {i+1}" for i in range(len(hidden))],
        xticklabels=False,   # too many hidden units to label individually
        ax=ax,
        cbar_kws={"label": "Activation"},
    )
    ax.set_xlabel("Hidden Units")
    ax.set_ylabel("Pitch in Sequence")
    ax.set_title("Hidden State Activations per Pitch\n(each row = model state after that pitch)")
    plt.tight_layout()

    if save_dir is not None:
        _save_fig(fig, save_dir, "hidden_state_heatmap")
    plt.show()


def _plot_average_delta_profile(all_deltas, save_dir):
    """
    Pads all delta vectors to the same length, then plots the mean ± std
    across all sampled at-bats. Shows at which pitch position the model
    typically changes its internal state the most.
    """
    max_len = max(len(d) for d in all_deltas)
    padded  = np.full((len(all_deltas), max_len), np.nan)
    for i, d in enumerate(all_deltas):
        padded[i, :len(d)] = d

    mean_delta = np.nanmean(padded, axis=0)
    std_delta  = np.nanstd(padded,  axis=0)
    xs         = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, mean_delta, color="#c0392b", linewidth=2, label="Mean Δ")
    ax.fill_between(
        xs,
        mean_delta - std_delta,
        mean_delta + std_delta,
        alpha=0.25, color="#c0392b", label="±1 std"
    )
    ax.set_xlabel("Pitch-to-Pitch Transition (e.g. 1 = pitch 1→2)")
    ax.set_ylabel("Mean L2 Hidden State Change")
    ax.set_title("Average Hidden State Change Profile Across At-Bats")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir is not None:
        _save_fig(fig, save_dir, "avg_delta_profile")
    plt.show()


def _plot_pca_trajectory_overlay(all_hidden, all_labels, save_dir):
    """
    Fits a single PCA across all AB hidden states and overlays each AB's
    trajectory as a faint line, making it easy to spot common 'paths'.
    """
    all_steps = np.vstack(all_hidden)
    pca       = PCA(n_components=2)
    pca.fit(all_steps)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = cm.tab10(np.linspace(0, 1, len(all_hidden)))

    for i, (h_ab, labels) in enumerate(zip(all_hidden, all_labels)):
        coords = pca.transform(h_ab)
        ax.plot(coords[:, 0], coords[:, 1], color=palette[i], alpha=0.5, linewidth=1.2)
        ax.scatter(coords[0, 0],  coords[0, 1],  color=palette[i], marker="o", s=60, zorder=5)
        ax.scatter(coords[-1, 0], coords[-1, 1], color=palette[i], marker="X", s=80, zorder=5)

    # Legend for markers
    ax.scatter([], [], marker="o", color="grey", label="First pitch")
    ax.scatter([], [], marker="X", color="grey", label="Last pitch")
    ax.legend(fontsize=9)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("PCA Trajectory Overlay — Multiple At-Bats\n(○ = start, ✕ = end)")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_dir is not None:
        _save_fig(fig, save_dir, "pca_trajectory_overlay")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, save_dir, stem):
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_analyses(emb_dims, num_layers, save_dir="evaluations/pitch_rnn/interpretability"):
    """
    Loads the latest model + test tensors and runs all three analyses.

    Parameters
    ----------
    emb_dims   : dict — must match what the model was trained with
    num_layers : int  — must match what the model was trained with
    save_dir   : base directory; a timestamped subfolder is created per run
    """
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir) / ts
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE   = Path(__file__).parent.parent.parent

    vocab_path   = get_latest_file(BASE / "model_shared" / "vocab",              "rnn_vocab_*.json")
    model_path   = get_latest_file(BASE / "model_shared" / "trained-parameters", "pitch_rnn_*.pt")
    tensors_path = get_latest_file(BASE / "model_shared" / "test_data",          "test_tensors_*.pt")

    model, cat_vocabs, y_vocab, id_to_pitch, feature_spec, num_classes = load_model_and_vocabs(
        vocab_path=vocab_path,
        model_path=model_path,
        emb_dims=emb_dims,
        num_layers=num_layers,
    )
    model = model.to(device)

    test_loader = load_test_loader(tensors_path)

    # ── 1. Permutation Importance ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("1. PERMUTATION IMPORTANCE")
    print("="*60)
    importance = permutation_importance(
        model, test_loader, device, feature_spec,
        n_repeats=3, save_dir=save_dir
    )
    print("\nRanked features:")
    for feat, drop in importance.items():
        print(f"  {feat:25s}  {drop:+.3f}%")

    # ── 2. Embedding Analysis ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. EMBEDDING ANALYSIS")
    print("="*60)
    emb_results = analyze_all_small_embeddings(
        model, cat_vocabs, feature_spec, save_dir=save_dir
    )

    # ── 3. Hidden State Analysis ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("3. HIDDEN STATE ANALYSIS")
    print("="*60)

    # Batch analysis: average delta profile + trajectory overlay
    all_hidden, all_deltas = analyze_hidden_states_batch(
        model, test_loader, device,
        n_abs=200,
        id_to_pitch=id_to_pitch,
        save_dir=save_dir,
    )

    # Single AB deep-dive: grab the first AB from the test set
    for x_cat, x_num, y, *_ in test_loader:
        real_mask = (y[0] != PAD_ID)
        real_steps = int(real_mask.sum())
        if real_steps >= 2:
            x_cat_ab = x_cat[0, :real_steps]
            x_num_ab = x_num[0, :real_steps]
            pitch_labels = [id_to_pitch.get(int(y[0, t].item()), "?") for t in range(real_steps)]
            print(f"\nSingle AB deep-dive — {real_steps} pitches: {pitch_labels}")
            analyze_hidden_states_single_ab(
                model, x_cat_ab, x_num_ab, device,
                pitch_labels=pitch_labels,
                save_dir=save_dir,
            )
            break

    print(f"\nAll plots saved to: {save_dir}")
    return importance, emb_results, all_hidden, all_deltas


if __name__ == "__main__":
    # ── Configure these to match your trained model ───────────────────────────
    EMB_DIMS = {
        "pitcher": 32,
        "batter":  32,
        "stand": 4,
        "p_throws": 4,
        "inning_topbot": 4,
        "count_state": 8,
        "prev_pitch_type": 16,
        "count_situation": 4,
        "prev_horiz_bucket": 4,
        "prev_vert_bucket": 4,
    }
    NUM_LAYERS = 2

    run_all_analyses(EMB_DIMS, NUM_LAYERS)