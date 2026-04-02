"""
visualize_results.py
Visualization and decision gate for TRIBE v2 Validation Experiment.

Reads results_summary.csv (or results_summary_dryrun.csv) and produces:
  1. ASCII comparison table (A vs B vs C)
  2. Bar chart  -- emotion accuracy per condition x category
  3. VAD correlation bar chart -- per condition
  4. Heatmap    -- per-category VAD correlations (A vs B vs C)
  5. decision_gate.txt -- conclusion text

Usage:
  python visualize_results.py                    # reads results_summary.csv
  python visualize_results.py --dry              # reads results_summary_dryrun.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--dry", action="store_true", help="Use dry-run results")
args = parser.parse_args()

suffix      = "_dryrun" if args.dry else ""
csv_path    = Path(f"results_summary{suffix}.csv")
json_path   = Path(f"results{suffix}.json")
out_dir     = Path("plots")
gate_path   = Path("decision_gate.txt")

if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Run evaluate.py first.")
    sys.exit(1)

out_dir.mkdir(exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(csv_path)
# Normalise column names to internal convention
df = df.rename(columns={
    "emotion_accuracy": "accuracy",
    "valence_r":   "vad_v_corr",
    "arousal_r":   "vad_a_corr",
    "dominance_r": "vad_d_corr",
})
print(f"Loaded {len(df)} rows from {csv_path}")
print(df.to_string(index=False))

# Expected columns: condition, category, n, accuracy, vad_v_corr, vad_a_corr, vad_d_corr
CONDITIONS  = ["A", "B", "C"]
CATEGORIES  = ["overall"] + [c for c in df["category"].unique() if c != "overall"]
COND_LABELS = {"A": "Text only", "B": "Text + Neural", "C": "Text + Shuffled"}
COND_COLORS = {"A": "#4878d0", "B": "#ee854a", "C": "#6acc65"}


# ── Helper ────────────────────────────────────────────────────────────────────

def get(cond: str, cat: str, col: str) -> float:
    row = df[(df["condition"] == cond) & (df["category"] == cat)]
    if row.empty or col not in row.columns:
        return float("nan")
    return float(row[col].iloc[0])


# ── 1. ASCII table ────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("RESULTS SUMMARY")
print("=" * 72)

header = f"{'Category':<14}  {'Metric':<16}  {'A (baseline)':>14}  {'B (neural)':>12}  {'C (shuffled)':>13}"
print(header)
print("-" * 72)

for cat in CATEGORIES:
    for metric, label in [("accuracy", "Accuracy"), ("vad_v_corr", "VAD-V corr"), ("vad_a_corr", "VAD-A corr")]:
        vals = {c: get(c, cat, metric) for c in CONDITIONS}
        a, b, cx = vals["A"], vals["B"], vals["C"]
        flag = ""
        if not any(np.isnan(v) for v in [a, b, cx]):
            if b > a > cx:
                flag = " <-- B>A>C [SIGNAL]"
            elif b > cx and b > a:
                flag = " <-- B best"
            elif b <= cx:
                flag = " <-- B<=C [NOISE]"
        row = f"{cat:<14}  {label:<16}  {a:>14.3f}  {b:>12.3f}  {cx:>13.3f}{flag}"
        print(row)
    print()


# ── 2. Accuracy bar chart ─────────────────────────────────────────────────────

cats_plot = [c for c in CATEGORIES if c != "overall"]
x = np.arange(len(cats_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, cond in enumerate(CONDITIONS):
    vals = [get(cond, cat, "accuracy") for cat in cats_plot]
    ax.bar(x + i * width, vals, width, label=COND_LABELS[cond],
           color=COND_COLORS[cond], edgecolor="white")

ax.set_xticks(x + width)
ax.set_xticklabels([c.capitalize() for c in cats_plot])
ax.set_ylabel("Emotion Accuracy")
ax.set_title("Emotion Accuracy by Category and Condition")
ax.set_ylim(0, 1.0)
ax.legend()
ax.axhline(1/6, color="gray", linestyle="--", linewidth=0.8, label="Chance (1/6)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
acc_path = out_dir / "accuracy_by_category.png"
plt.savefig(acc_path, dpi=150)
plt.close()
print(f"Saved: {acc_path}")


# ── 3. VAD correlation bar chart (overall) ────────────────────────────────────

vad_dims   = ["vad_v_corr", "vad_a_corr", "vad_d_corr"]
vad_labels = ["Valence", "Arousal", "Dominance"]
x2 = np.arange(len(vad_dims))

fig, ax = plt.subplots(figsize=(8, 5))
for i, cond in enumerate(CONDITIONS):
    vals = [get(cond, "overall", dim) for dim in vad_dims]
    ax.bar(x2 + i * width, vals, width, label=COND_LABELS[cond],
           color=COND_COLORS[cond], edgecolor="white")

ax.set_xticks(x2 + width)
ax.set_xticklabels(vad_labels)
ax.set_ylabel("Pearson r")
ax.set_title("VAD Correlation (overall) by Condition")
ax.axhline(0, color="black", linewidth=0.8)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
vad_path = out_dir / "vad_correlation_overall.png"
plt.savefig(vad_path, dpi=150)
plt.close()
print(f"Saved: {vad_path}")


# ── 4. Heatmap: VAD-V correlation per category x condition ───────────────────

heat_cats  = [c for c in CATEGORIES if c != "overall"]
heat_data  = np.array([[get(cond, cat, "vad_v_corr") for cond in CONDITIONS]
                        for cat in heat_cats])

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9)
ax.set_yticks(range(len(heat_cats)))
ax.set_yticklabels([c.capitalize() for c in heat_cats])
ax.set_title("VAD-Valence Pearson r\nby Category and Condition")
plt.colorbar(im, ax=ax, label="Pearson r")
for i in range(len(heat_cats)):
    for j in range(len(CONDITIONS)):
        v = heat_data[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
plt.tight_layout()
heat_path = out_dir / "vad_valence_heatmap.png"
plt.savefig(heat_path, dpi=150)
plt.close()
print(f"Saved: {heat_path}")


# ── 5. Decision gate ──────────────────────────────────────────────────────────

ov_acc_a  = get("A", "overall", "accuracy")
ov_acc_b  = get("B", "overall", "accuracy")
ov_acc_c  = get("C", "overall", "accuracy")
ov_vad_a  = get("A", "overall", "vad_v_corr")
ov_vad_b  = get("B", "overall", "vad_v_corr")
ov_vad_c  = get("C", "overall", "vad_v_corr")

impl_acc_a = get("A", "implicit", "accuracy")
impl_acc_b = get("B", "implicit", "accuracy")
impl_acc_c = get("C", "implicit", "accuracy")

b_beats_a_acc  = ov_acc_b  > ov_acc_a
b_beats_c_acc  = ov_acc_b  > ov_acc_c
a_beats_c_acc  = ov_acc_a  > ov_acc_c
b_beats_a_vad  = ov_vad_b  > ov_vad_a
b_beats_c_vad  = ov_vad_b  > ov_vad_c
impl_b_best    = impl_acc_b > impl_acc_a and impl_acc_b > impl_acc_c

full_signal    = b_beats_a_acc and b_beats_c_acc and a_beats_c_acc
partial_signal = (b_beats_a_acc or b_beats_a_vad) and b_beats_c_acc

if full_signal:
    verdict = "VALIDATED"
    detail  = ("B > A > C on both accuracy and VAD. Neural context carries "
               "real emotional signal. Recommend building the full system.")
elif partial_signal:
    verdict = "PARTIAL SIGNAL"
    detail  = ("B outperforms C and improves over A on at least one metric. "
               "Signal is present but not consistent across all dimensions. "
               "Investigate formatter and ROI weighting before full build.")
elif b_beats_c_acc and not b_beats_a_acc:
    verdict = "FORMATTER ISSUE"
    detail  = ("B > C but B <= A. Neural context adds some signal above "
               "shuffled noise but underperforms text-only baseline. "
               "The brain-to-text formatter may be losing information. "
               "Revisit ROI selection and z-score normalization.")
else:
    verdict = "NO SIGNAL DETECTED"
    detail  = ("B does not consistently outperform C. Neural context is "
               "indistinguishable from shuffled noise. Either TRIBE v2 "
               "predictions are not carrying emotion signal for this text "
               "domain, or the evaluation pipeline needs revision.")

lines = [
    "TRIBE v2 Validation Experiment -- Decision Gate",
    "=" * 50,
    "",
    f"Verdict: {verdict}",
    "",
    "Overall metrics:",
    f"  Accuracy  -- A: {ov_acc_a:.3f}  B: {ov_acc_b:.3f}  C: {ov_acc_c:.3f}",
    f"  VAD-V r   -- A: {ov_vad_a:.3f}  B: {ov_vad_b:.3f}  C: {ov_vad_c:.3f}",
    "",
    "Implicit category (key test):",
    f"  Accuracy  -- A: {impl_acc_a:.3f}  B: {impl_acc_b:.3f}  C: {impl_acc_c:.3f}",
    "",
    "Analysis:",
    detail,
    "",
    "Plots saved to: plots/",
    f"  {acc_path.name}",
    f"  {vad_path.name}",
    f"  {heat_path.name}",
]

gate_text = "\n".join(lines)
gate_path.write_text(gate_text, encoding="utf-8")

print("\n" + "=" * 72)
print(gate_text)
print("=" * 72)
print(f"\nSaved decision gate -> {gate_path}")
