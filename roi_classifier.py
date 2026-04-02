"""
roi_classifier.py
Direct ROI Classifier for TRIBE v2 Validation Experiment

Trains classifiers directly on the 7 ROI mean activations (no LLM).
Answers: "How much discriminative emotion signal exists in the raw TRIBE v2 activations?"

Runs on both text-based and emotional audio activations, then compares.

Outputs:
  plots/roi_classifier_comparison.png   -- accuracy comparison bar chart
  plots/roi_emotion_fingerprint.png     -- heatmap: 4 emotions x 7 ROIs (z-scored means)
  roi_classifier_results.txt            -- text summary
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

# ── Config ────────────────────────────────────────────────────────────────────

ROI_NAMES = ["vmPFC", "amygdala", "insula", "ACC", "TPJ", "hippocampus", "motor"]
EMOTIONS   = ["happy", "angry", "sad", "fearful"]
CATEGORIES = ["explicit", "implicit", "sarcastic"]
N_FOLDS    = 5
RANDOM_STATE = 42

Path("plots").mkdir(exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(activations_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        X          -- (n_samples, 7) float array of ROI means
        y_emotion  -- (n_samples,) int array of emotion label indices
        y_category -- (n_samples,) str array of category labels
        ids        -- list of sample IDs
    """
    samples = json.load(open("mosei_samples.json", encoding="utf-8"))
    activations = json.load(open(activations_path, encoding="utf-8"))

    # Index by id
    sample_map = {s["id"]: s for s in samples}
    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

    X, y_emotion, y_category, ids = [], [], [], []
    for act in activations:
        sid = act["id"]
        s = sample_map[sid]
        roi = act["roi_means"]
        X.append([roi[r] for r in ROI_NAMES])
        y_emotion.append(emotion_to_idx.get(s["emotion_label"], -1))
        y_category.append(s["category"])
        ids.append(sid)

    X = np.array(X, dtype=np.float32)
    y_emotion = np.array(y_emotion, dtype=np.int32)
    y_category = np.array(y_category)
    return X, y_emotion, y_category, ids


# ── Classification ────────────────────────────────────────────────────────────

def run_classifiers(X: np.ndarray, y: np.ndarray, label: str) -> dict[str, float]:
    """
    Runs 3 classifiers with stratified 5-fold CV.
    Returns dict: classifier_name -> mean accuracy.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "SVM":    SVC(kernel="rbf", random_state=RANDOM_STATE),
        "RF":     RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print(f"\n  {label}")
    print(f"  {'Classifier':<12} {'Acc (5-fold CV)':>16} {'Chance':>8}")
    print(f"  {'-'*40}")

    chance = 1.0 / len(EMOTIONS)
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        results[name] = mean_acc
        flag = " *" if mean_acc > chance + 0.05 else ""
        print(f"  {name:<12} {mean_acc:>14.1%}   {chance:>6.1%}{flag}")

    return results


def run_per_category(
    X: np.ndarray,
    y: np.ndarray,
    y_cat: np.ndarray,
    label: str,
) -> dict[str, float]:
    """
    Runs LogReg per category subset.
    Returns dict: category -> mean accuracy.
    """
    scaler = StandardScaler()
    results = {}

    print(f"\n  {label} -- per category (LogReg, 5-fold CV)")
    print(f"  {'Category':<12} {'N':>4} {'Acc':>8} {'Chance':>8}")
    print(f"  {'-'*36}")

    for cat in CATEGORIES:
        mask = y_cat == cat
        Xc, yc = X[mask], y[mask]
        if len(np.unique(yc)) < 2 or len(yc) < N_FOLDS:
            print(f"  {cat:<12} {mask.sum():>4}  too few samples")
            continue

        Xc_scaled = scaler.fit_transform(Xc)
        cv = StratifiedKFold(n_splits=min(N_FOLDS, len(yc)), shuffle=True, random_state=RANDOM_STATE)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        scores = cross_val_score(clf, Xc_scaled, yc, cv=cv, scoring="accuracy")
        mean_acc = scores.mean()
        results[cat] = mean_acc
        n_classes = len(np.unique(yc))
        chance = 1.0 / n_classes
        flag = " *" if mean_acc > chance + 0.05 else ""
        print(f"  {cat:<12} {mask.sum():>4} {mean_acc:>7.1%}  {chance:>6.1%}{flag}")

    return results


# ── Emotion fingerprint heatmap ───────────────────────────────────────────────

def emotion_fingerprint(
    X: np.ndarray,
    y: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Heatmap of z-scored mean ROI activations per emotion class.
    Rows = emotions, columns = ROIs.
    """
    scaler = StandardScaler()
    X_z = scaler.fit_transform(X)

    matrix = np.zeros((len(EMOTIONS), len(ROI_NAMES)))
    for i, emo in enumerate(EMOTIONS):
        mask = y == i
        if mask.sum() > 0:
            matrix[i] = X_z[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(ROI_NAMES)))
    ax.set_xticklabels(ROI_NAMES, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=10)

    # Annotate cells
    for i in range(len(EMOTIONS)):
        for j in range(len(ROI_NAMES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label="z-scored mean activation")
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Comparison bar chart ──────────────────────────────────────────────────────

def comparison_chart(
    results_text: dict[str, float],
    results_emo: dict[str, float],
    save_path: str,
) -> None:
    """
    Side-by-side bar chart of classifier accuracy: text-based vs emotional audio.
    """
    clf_names = list(results_text.keys())
    x = np.arange(len(clf_names))
    width = 0.32
    chance = 1.0 / len(EMOTIONS)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, [results_text[c] for c in clf_names],
                   width, label="Text-based", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, [results_emo[c] for c in clf_names],
                   width, label="Emotional Audio", color="#DD8452", alpha=0.85)

    ax.axhline(chance, color="red", linestyle="--", linewidth=1.2, label=f"Chance ({chance:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, fontsize=11)
    ax.set_ylabel("Accuracy (5-fold CV)")
    ax.set_ylim(0, 0.6)
    ax.set_title("ROI Classifier Accuracy: Text vs Emotional Audio\n(TRIBE v2 activations, 4-class emotion, n=100)", fontsize=11)
    ax.legend(fontsize=9)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance(X: np.ndarray, y: np.ndarray, label: str) -> None:
    """Prints LogReg coefficient norms per ROI (multi-class l2)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_scaled, y)
    # For multi-class, coef_ is (n_classes, n_features) -- take L2 norm
    importance = np.linalg.norm(clf.coef_, axis=0)
    ranked = sorted(zip(ROI_NAMES, importance), key=lambda x: x[1], reverse=True)
    print(f"\n  ROI importance ({label}, LogReg coef L2 norm):")
    for roi, imp in ranked:
        bar = "#" * int(imp * 40 / max(importance))
        print(f"    {roi:<14} {imp:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Direct ROI Classifier -- TRIBE v2 Validation Experiment")
    print("=" * 60)

    # Load both activation sets
    print("\n[1/5] Loading data...")
    X_text, y_text, ycat_text, _ = load_data("tribe_activations.json")
    X_emo,  y_emo,  ycat_emo,  _ = load_data("tribe_activations_emotional.json")
    print(f"  Text-based : {X_text.shape[0]} samples x {X_text.shape[1]} ROIs")
    print(f"  Emotional  : {X_emo.shape[0]} samples x {X_emo.shape[1]} ROIs")

    chance = 1.0 / len(EMOTIONS)
    print(f"  Chance level: {chance:.1%} (4-class uniform)")

    # Classifier accuracy -- all samples
    print("\n[2/5] Running classifiers (5-fold CV, all samples)...")
    results_text = run_classifiers(X_text, y_text, "Text-based activations")
    results_emo  = run_classifiers(X_emo,  y_emo,  "Emotional audio activations")

    # Per-category breakdown
    print("\n[3/5] Per-category breakdown (LogReg)...")
    run_per_category(X_text, y_text, ycat_text, "Text-based")
    run_per_category(X_emo,  y_emo,  ycat_emo,  "Emotional audio")

    # Feature importance
    print("\n[4/5] Feature importance...")
    feature_importance(X_text, y_text, "text-based")
    feature_importance(X_emo,  y_emo,  "emotional audio")

    # Plots
    print("\n[5/5] Generating plots...")
    comparison_chart(results_text, results_emo, "plots/roi_classifier_comparison.png")
    emotion_fingerprint(X_text, y_text,
                        "Emotion Activation Fingerprint -- Text-based TRIBE v2",
                        "plots/roi_emotion_fingerprint_text.png")
    emotion_fingerprint(X_emo, y_emo,
                        "Emotion Activation Fingerprint -- Emotional Audio TRIBE v2",
                        "plots/roi_emotion_fingerprint_emotional.png")

    # Text summary
    lines = [
        "ROI Classifier Results -- TRIBE v2 Validation Experiment",
        "=" * 60,
        f"Chance level: {chance:.1%} (4-class, 100 samples)",
        "",
        "Text-based activations:",
    ]
    for clf, acc in results_text.items():
        lines.append(f"  {clf:<10} {acc:.1%}")
    lines += ["", "Emotional audio activations:"]
    for clf, acc in results_emo.items():
        lines.append(f"  {clf:<10} {acc:.1%}")

    best_text = max(results_text.values())
    best_emo  = max(results_emo.values())
    lines += [
        "",
        f"Best text-based : {best_text:.1%}  ({'above' if best_text > chance + 0.05 else 'at/near'} chance)",
        f"Best emotional  : {best_emo:.1%}  ({'above' if best_emo  > chance + 0.05 else 'at/near'} chance)",
        "",
        "Interpretation:",
        "  If classifiers exceed chance, TRIBE v2 activations encode emotion.",
        "  If emotional audio > text-based, prosody enriches the brain signal.",
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    Path("roi_classifier_results.txt").write_text(summary, encoding="utf-8")
    print("\nSaved -> roi_classifier_results.txt")
    print("Saved -> plots/roi_classifier_comparison.png")
    print("Saved -> plots/roi_emotion_fingerprint_text.png")
    print("Saved -> plots/roi_emotion_fingerprint_emotional.png")


if __name__ == "__main__":
    main()
