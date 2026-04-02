"""
brain_formatter.py
Brain-to-Text Formatter for TRIBE v2 Validation Experiment

Pipeline:
  1. Load tribe_activations.json (100 samples with per-ROI means/stds)
  2. Z-score each ROI mean across the full batch of 100 samples
  3. Map z-scores to qualitative levels and natural language descriptions
  4. Generate real_context (matched text + activations) and
     shuffled_context (text paired with a DIFFERENT sample's activations)
  5. Save formatted_contexts.json

Z-score → level mapping:
  z >  1.5          → "high"
  z  0.5 – 1.5      → "moderate"
  z -0.5 – 0.5      → "neutral"
  z -1.5 – -0.5     → "low"
  z < -1.5          → "very low"
"""

import json
import os
import random
from typing import Any

import numpy as np

random.seed(99)
np.random.seed(99)

# ─── ROI metadata ─────────────────────────────────────────────────────────────

# Vertex count matches TRIBE v2 fsaverage5 output (2 x 10,242 vertices)
N_VERTICES: int = 20_484

ROI_META: dict[str, dict[str, Any]] = {
    "vmPFC": {
        "label": "valence/reward",
        "descriptions": {
            "very low": "strongly negative valence, suppressed reward circuitry",
            "low":      "mild negative valence, reduced reward response",
            "neutral":  "neutral valence, baseline reward state",
            "moderate": "mild positive valence, moderate reward engagement",
            "high":     "strong positive valence signal",
        },
    },
    "amygdala": {
        "label": "threat/arousal",
        "descriptions": {
            "very low": "suppressed threat response, minimal emotional arousal",
            "low":      "low threat detection, relatively calm state",
            "neutral":  "baseline threat monitoring and arousal",
            "moderate": "elevated arousal and notable threat response",
            "high":     "heightened threat detection and strong emotional arousal",
        },
    },
    "insula": {
        "label": "interoception/disgust",
        "descriptions": {
            "very low": "very low visceral response, detached bodily state",
            "low":      "low interoceptive signal, mild bodily calm",
            "neutral":  "neutral bodily state",
            "moderate": "notable interoceptive activity, mild disgust or discomfort",
            "high":     "strong visceral feeling, heightened disgust response",
        },
    },
    "ACC": {
        "label": "conflict/uncertainty",
        "descriptions": {
            "very low": "minimal emotional conflict, very high certainty",
            "low":      "low emotional conflict, moderate certainty",
            "neutral":  "baseline conflict monitoring",
            "moderate": "moderate emotional ambivalence and conflict",
            "high":     "high emotional ambivalence and strong conflict signal",
        },
    },
    "TPJ": {
        "label": "social cognition",
        "descriptions": {
            "very low": "minimal social processing, very low perspective-taking",
            "low":      "reduced interpersonal engagement",
            "neutral":  "baseline social monitoring",
            "moderate": "moderate interpersonal processing",
            "high":     "strong interpersonal processing, active perspective-taking",
        },
    },
    "hippocampus": {
        "label": "memory/context",
        "descriptions": {
            "very low": "minimal contextual grounding, low memory retrieval",
            "low":      "low autobiographical context activation",
            "neutral":  "baseline contextual processing",
            "moderate": "moderate autobiographical context and memory encoding",
            "high":     "strong memory encoding and rich contextual grounding",
        },
    },
    "motor": {
        "label": "embodied simulation",
        "descriptions": {
            "very low": "very low action readiness, minimal embodied simulation",
            "low":      "low action readiness",
            "neutral":  "baseline motor simulation",
            "moderate": "moderate embodied simulation and action preparation",
            "high":     "high action readiness and strong embodied simulation",
        },
    },
}

# ─── Z-score → qualitative level ─────────────────────────────────────────────

def z_to_level(z: float) -> str:
    """
    Map a z-score to a qualitative activation level.

        z >  1.5          -> 'high'
        z  0.5 to  1.5    -> 'moderate'
        z -0.5 to  0.5    -> 'neutral'
        z -1.5 to -0.5    -> 'low'
        z < -1.5          -> 'very low'
    """
    if z > 1.5:
        return "high"
    elif z > 0.5:
        return "moderate"
    elif z >= -0.5:
        return "neutral"
    elif z >= -1.5:
        return "low"
    else:
        return "very low"


def z_to_clarity(z: float) -> str:
    """
    Map z-score magnitude to a confidence/clarity tag for the output block.

        |z| > 1.5   -> '[clear]'      (strong, reliable signal)
        |z| > 0.5   -> '[moderate]'   (moderate signal)
        otherwise   -> '[uncertain]'  (weak / ambiguous signal)
    """
    abs_z = abs(z)
    if abs_z > 1.5:
        return "[clear]"
    elif abs_z > 0.5:
        return "[moderate]"
    else:
        return "[uncertain]"


# ─── Natural language context block builder ───────────────────────────────────

def derive_vad_labels(z_scores: dict[str, float]) -> tuple[str, str]:
    """
    Derive coarse Valence and Arousal labels from ROI z-scores.

    Uses vmPFC z-score for Valence and amygdala z-score for Arousal,
    matching the standard neuroscientific interpretation of these regions.

    Returns:
        valence_label: 'positive' | 'neutral' | 'negative'
        arousal_label: 'high' | 'moderate' | 'low'
    """
    v_z = z_scores["vmPFC"]
    a_z = z_scores["amygdala"]

    if v_z > 0.5:
        valence_label = "positive"
    elif v_z < -0.5:
        valence_label = "negative"
    else:
        valence_label = "neutral"

    if a_z > 0.5:
        arousal_label = "high"
    elif a_z < -0.5:
        arousal_label = "low"
    else:
        arousal_label = "moderate"

    return valence_label, arousal_label


def build_context_block(z_scores: dict[str, float]) -> str:
    """
    Build the structured [Neural Emotion Context] natural language block
    from a dict of per-ROI z-scores.

    Args:
        z_scores: dict mapping ROI name -> z-scored activation value

    Returns:
        Multi-line string in the standard TRIBE v2 context format.
    """
    lines: list[str] = ["[Neural Emotion Context]"]

    for roi_name, meta in ROI_META.items():
        z = z_scores[roi_name]
        level = z_to_level(z)
        clarity = z_to_clarity(z)
        description = meta["descriptions"][level]
        label = meta["label"]
        lines.append(f"- {roi_name} ({label}): {description} {clarity}")

    valence_label, arousal_label = derive_vad_labels(z_scores)
    lines.append(
        f"Derived VAD estimate: Valence={valence_label}, Arousal={arousal_label}"
    )

    return "\n".join(lines)


# ─── Derangement helper ───────────────────────────────────────────────────────

def make_derangement(n: int) -> list[int]:
    """
    Return a permutation of [0, n-1] with no fixed points (derangement).

    Used to pair each sample's text with a DIFFERENT sample's activations
    for Condition C (shuffled context). Guarantees no sample is ever
    paired with its own activations.
    """
    indices = list(range(n))
    while True:
        shuffled = indices[:]
        random.shuffle(shuffled)
        if all(shuffled[i] != i for i in range(n)):
            return shuffled


# ─── Z-scoring ────────────────────────────────────────────────────────────────

def compute_z_scores(
    records: list[dict[str, Any]]
) -> list[dict[str, float]]:
    """
    Z-score each ROI's mean activation across the full batch of 100 samples.

    For each ROI:
        z_i = (x_i - mean(X)) / std(X)   where X = all 100 samples' ROI means

    Args:
        records: list of dicts from tribe_activations.json

    Returns:
        List of per-sample z-score dicts, one per record, same order.
    """
    roi_names = list(ROI_META.keys())

    # Collect raw means per ROI
    raw: dict[str, list[float]] = {roi: [] for roi in roi_names}
    for rec in records:
        for roi in roi_names:
            raw[roi].append(rec["roi_means"][roi])

    # Compute batch statistics
    roi_mean: dict[str, float] = {}
    roi_std: dict[str, float] = {}
    for roi in roi_names:
        arr = np.array(raw[roi])
        roi_mean[roi] = float(arr.mean())
        roi_std[roi] = float(arr.std())
        if roi_std[roi] < 1e-8:
            roi_std[roi] = 1e-8   # avoid division by zero

    # Z-score each sample
    z_scores_per_sample: list[dict[str, float]] = []
    for rec in records:
        z: dict[str, float] = {}
        for roi in roi_names:
            raw_val = rec["roi_means"][roi]
            z[roi] = (raw_val - roi_mean[roi]) / roi_std[roi]
        z_scores_per_sample.append(z)

    return z_scores_per_sample


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Load activations, format to natural language, save formatted_contexts.json."""
    print("=" * 60)
    print("Brain-to-Text Formatter -- TRIBE v2 Validation")
    print("=" * 60)

    # ── Load activations ──────────────────────────────────────────────────────
    input_path = "tribe_activations_emotional.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Run tribe_inference_emotional.py first."
        )
    with open(input_path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    print(f"\n[1/4] Loaded {len(records)} activation records from {input_path}")

    # ── Z-score ───────────────────────────────────────────────────────────────
    print("[2/4] Z-scoring ROI means across the full batch...")
    z_scores_per_sample = compute_z_scores(records)
    print(f"      Done. Z-scored {len(z_scores_per_sample)} samples x "
          f"{len(ROI_META)} ROIs.")

    # ── Build real context blocks ─────────────────────────────────────────────
    print("[3/4] Building natural language context blocks...")
    real_contexts: list[str] = [
        build_context_block(z) for z in z_scores_per_sample
    ]

    # ── Build shuffled context blocks (Condition C) ───────────────────────────
    # Each sample's text is paired with a DIFFERENT sample's activations.
    # We use a derangement to guarantee no sample is paired with itself.
    derangement = make_derangement(len(records))
    shuffled_contexts: list[str] = [
        build_context_block(z_scores_per_sample[derangement[i]])
        for i in range(len(records))
    ]

    # ── Save ──────────────────────────────────────────────────────────────────
    output: list[dict[str, str]] = []
    for i, rec in enumerate(records):
        output.append({
            "id":               rec["id"],
            "real_context":     real_contexts[i],
            "shuffled_context": shuffled_contexts[i],
        })

    output_path = "formatted_contexts_emotional.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[4/4] Saved {len(output)} formatted context pairs -> {output_path}")

    # ── Preview: 5 samples ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREVIEW -- first 5 sample IDs and their real context")
    print("=" * 60)
    for entry in output[:5]:
        print(f"\nID: {entry['id']}")
        print(entry["real_context"])

    # Full real + shuffled print for sample index 0
    print("\n" + "=" * 60)
    print("FULL COMPARISON (real vs shuffled) -- sample 0")
    print("=" * 60)

    sample_0_id    = output[0]["id"]
    sample_0_text  = records[0]["text"]
    shuffled_src   = records[derangement[0]]["id"]

    print(f"\nSample ID   : {sample_0_id}")
    print(f"Text        : {sample_0_text[:100]}")
    print(f"Shuffled src: {shuffled_src}  "
          f"(activations taken from this sample's brain data)\n")

    print("--- REAL CONTEXT ---")
    print(output[0]["real_context"])

    print("\n--- SHUFFLED CONTEXT (Condition C) ---")
    print(output[0]["shuffled_context"])


if __name__ == "__main__":
    main()
