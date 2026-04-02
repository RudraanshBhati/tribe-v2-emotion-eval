"""
evaluate.py
LLM Evaluation Script for TRIBE v2 Validation Experiment

Runs 100 samples through 3 conditions using llama3.1:8b via Ollama:
  Condition A — Text only
  Condition B — Text + real neural context (TRIBE v2 activations)
  Condition C — Text + shuffled neural context (mismatched activations)

Metrics computed:
  - Emotion label accuracy  (overall + per category)
  - VAD Pearson correlation (overall + per category)

Outputs:
  results.json         — all raw per-sample results
  results_summary.csv  — clean metrics table

IMPORTANT: Run with DRY_RUN=True (default) to test on 10 samples first.
           Set DRY_RUN=False after confirming to run the full 100.
"""

import csv
import json
import os
import re
import time
from typing import Any

import numpy as np
import ollama
from scipy.stats import pearsonr
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

DRY_RUN: bool = False         # Set to False for the full 100-sample run
DRY_RUN_N: int = 10           # Number of samples in dry run
MODEL: str = "llama3.1:8b"
SLEEP_BETWEEN_CALLS: float = 0.5   # seconds — avoids overloading context

VALID_EMOTIONS: set[str] = {
    "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"
}

CATEGORIES: list[str] = ["explicit", "implicit", "sarcastic"]

# ─── Prompt builders ──────────────────────────────────────────────────────────

JSON_SCHEMA: str = '{{"emotion": "", "valence": 0, "arousal": 0, "dominance": 0}}'

QUESTION: str = (
    "What is the primary emotion in this text?\n"
    "Choose exactly one: happy, sad, angry, fearful, disgusted, surprised, neutral.\n"
    "Also estimate valence (-3 to +3), arousal (0 to 3), dominance (0 to 3).\n"
    "Text: {text}\n"
    "Respond only in JSON with no extra text:\n"
    f"{JSON_SCHEMA}"
)


def prompt_a(text: str) -> str:
    """Condition A — text only."""
    return QUESTION.format(text=text)


def prompt_b(text: str, real_context: str) -> str:
    """Condition B — text + real neural context."""
    return f"{real_context}\n\n{QUESTION.format(text=text)}"


def prompt_c(text: str, shuffled_context: str) -> str:
    """Condition C — text + shuffled (mismatched) neural context."""
    return f"{shuffled_context}\n\n{QUESTION.format(text=text)}"


# ─── Ollama inference ─────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> dict[str, Any]:
    """
    Send a prompt to llama3.1:8b via the local Ollama server and return
    a parsed dict with keys: emotion, valence, arousal, dominance.

    Robustly handles:
      - Markdown code fences (```json ... ```)
      - Extra whitespace / trailing text
      - Completely malformed responses (returns null values)

    OLLAMA_NUM_GPU=1 is set in the environment before this module is
    imported to ensure GPU inference on the RTX 4070.
    """
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},   # deterministic outputs
        )
        raw: str = response["message"]["content"].strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # Extract first JSON object if extra text is present
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        parsed: dict[str, Any] = json.loads(raw)

        # Normalise emotion string
        emotion = str(parsed.get("emotion", "")).strip().lower()
        if emotion not in VALID_EMOTIONS:
            emotion = "neutral"   # safe fallback for unexpected labels

        return {
            "emotion":   emotion,
            "valence":   float(parsed.get("valence",   0.0)),
            "arousal":   float(parsed.get("arousal",   0.0)),
            "dominance": float(parsed.get("dominance", 0.0)),
        }

    except Exception:
        return {
            "emotion":   None,
            "valence":   None,
            "arousal":   None,
            "dominance": None,
        }


# ─── Metrics ──────────────────────────────────────────────────────────────────

def safe_pearsonr(x: list[float], y: list[float]) -> float:
    """
    Pearson correlation between two lists, filtering out None pairs.
    Returns 0.0 if fewer than 2 valid pairs exist.
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    r, _ = pearsonr(list(xs), list(ys))
    return float(r) if not np.isnan(r) else 0.0


def compute_metrics(
    results: list[dict[str, Any]],
    condition: str,
    category_filter: str | None = None,
) -> dict[str, float]:
    """
    Compute emotion accuracy and VAD Pearson correlations for one condition.

    Args:
        results:         full list of result dicts
        condition:       'A', 'B', or 'C'
        category_filter: if given, restrict to samples of that category

    Returns:
        dict with keys: emotion_accuracy, valence_r, arousal_r, dominance_r
    """
    subset = [
        r for r in results
        if (category_filter is None or r["category"] == category_filter)
    ]
    if not subset:
        return {"emotion_accuracy": 0.0, "valence_r": 0.0,
                "arousal_r": 0.0, "dominance_r": 0.0}

    pred_key = f"pred_{condition.lower()}"
    gt_emotions  = [r["gt_emotion"]   for r in subset]
    gt_valence   = [r["gt_valence"]   for r in subset]
    gt_arousal   = [r["gt_arousal"]   for r in subset]
    gt_dominance = [r["gt_dominance"] for r in subset]

    pred_emotions  = [r[pred_key]["emotion"]   for r in subset]
    pred_valence   = [r[pred_key]["valence"]   for r in subset]
    pred_arousal   = [r[pred_key]["arousal"]   for r in subset]
    pred_dominance = [r[pred_key]["dominance"] for r in subset]

    correct = sum(
        1 for gt, pr in zip(gt_emotions, pred_emotions)
        if gt == pr and pr is not None
    )
    accuracy = correct / len(subset)

    return {
        "emotion_accuracy": round(accuracy, 4),
        "valence_r":        round(safe_pearsonr(gt_valence,   pred_valence),   4),
        "arousal_r":        round(safe_pearsonr(gt_arousal,   pred_arousal),   4),
        "dominance_r":      round(safe_pearsonr(gt_dominance, pred_dominance), 4),
    }


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    samples: list[dict[str, Any]],
    contexts: dict[str, dict[str, str]],
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """
    Run all 3 conditions on every sample (or first DRY_RUN_N if dry_run=True).

    Args:
        samples:  list of dicts from mosei_samples.json
        contexts: dict mapping sample id -> {real_context, shuffled_context}
        dry_run:  if True, process only the first DRY_RUN_N samples

    Returns:
        list of result dicts, one per sample
    """
    subset = samples[:DRY_RUN_N] if dry_run else samples
    total_calls = len(subset) * 3
    results: list[dict[str, Any]] = []

    label = f"DRY RUN ({DRY_RUN_N} samples)" if dry_run else f"FULL RUN ({len(subset)} samples)"
    print(f"\n{'=' * 60}")
    print(f"Evaluation: {label}  |  {total_calls} LLM calls total")
    print(f"{'=' * 60}\n")

    correct_counts = {"A": 0, "B": 0, "C": 0}
    processed = 0

    pbar = tqdm(total=total_calls, desc="Evaluating", unit="call")

    for sample in subset:
        sid      = sample["id"]
        text     = sample["text"]
        category = sample["category"]
        ctx      = contexts[sid]

        gt_vad = sample["vad_scores"]

        result: dict[str, Any] = {
            "id":           sid,
            "text":         text,
            "category":     category,
            "gt_emotion":   sample["emotion_label"],
            "gt_valence":   gt_vad["valence"],
            "gt_arousal":   gt_vad["arousal"],
            "gt_dominance": gt_vad["dominance"],
        }

        for cond, prompt_fn, ctx_arg in [
            ("A", prompt_a, None),
            ("B", prompt_b, ctx["real_context"]),
            ("C", prompt_c, ctx["shuffled_context"]),
        ]:
            if ctx_arg is None:
                prompt = prompt_fn(text)
            else:
                prompt = prompt_fn(text, ctx_arg)

            pred = call_ollama(prompt)
            result[f"pred_{cond.lower()}"] = pred

            # Running accuracy for progress bar
            if pred["emotion"] == sample["emotion_label"]:
                correct_counts[cond] += 1

            processed_for_cond = processed + 1
            acc_a = correct_counts["A"] / max(processed_for_cond, 1)
            acc_b = correct_counts["B"] / max(processed_for_cond, 1)

            pbar.set_postfix({
                "sample": sid,
                "cond": cond,
                "acc_A": f"{acc_a:.0%}",
                "acc_B": f"{acc_b:.0%}",
            })
            pbar.update(1)
            time.sleep(SLEEP_BETWEEN_CALLS)

        processed += 1
        results.append(result)

    pbar.close()
    return results


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Load data, run evaluation, compute metrics, save outputs."""
    # Ensure Ollama uses GPU
    os.environ["OLLAMA_NUM_GPU"] = "1"

    print("=" * 60)
    print("LLM Evaluation -- TRIBE v2 Validation Experiment")
    print(f"Model  : {MODEL}")
    print(f"Mode   : {'DRY RUN (10 samples)' if DRY_RUN else 'FULL RUN (100 samples)'}")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────────
    for path in ("mosei_samples.json", "formatted_contexts_emotional.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run previous steps first.")

    with open("mosei_samples.json", encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)

    with open("formatted_contexts_emotional.json", encoding="utf-8") as f:
        ctx_list: list[dict[str, str]] = json.load(f)

    # Index contexts by sample id for O(1) lookup
    contexts: dict[str, dict[str, str]] = {c["id"]: c for c in ctx_list}
    print(f"\nLoaded {len(samples)} samples and {len(contexts)} context pairs.")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = run_evaluation(samples, contexts, dry_run=DRY_RUN)

    # ── Save raw results ──────────────────────────────────────────────────────
    suffix = "_dryrun" if DRY_RUN else "_emotional"
    results_path = f"results{suffix}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved -> {results_path}")

    # ── Compute and print metrics ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("METRICS SUMMARY")
    print(f"{'=' * 60}")

    summary_rows: list[dict[str, Any]] = []

    header = f"{'Condition':<10} {'Category':<12} {'Acc':>6} {'V_r':>7} {'A_r':>7} {'D_r':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    for condition in ["A", "B", "C"]:
        for category in (CATEGORIES + ["overall"]):
            cat_filter = None if category == "overall" else category
            m = compute_metrics(results, condition, cat_filter)

            row = {
                "condition":        condition,
                "category":         category,
                "emotion_accuracy": m["emotion_accuracy"],
                "valence_r":        m["valence_r"],
                "arousal_r":        m["arousal_r"],
                "dominance_r":      m["dominance_r"],
            }
            summary_rows.append(row)

            print(
                f"  {condition:<8} {category:<12} "
                f"{m['emotion_accuracy']:>6.1%} "
                f"{m['valence_r']:>7.3f} "
                f"{m['arousal_r']:>7.3f} "
                f"{m['dominance_r']:>7.3f}"
            )

        print()

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = f"results_summary{suffix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "category", "emotion_accuracy",
                        "valence_r", "arousal_r", "dominance_r"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV saved -> {csv_path}")

    if DRY_RUN:
        print(f"\n{'=' * 60}")
        print("DRY RUN COMPLETE.")
        print("Review the metrics above, then set DRY_RUN = False in")
        print("evaluate.py and re-run for the full 100-sample evaluation.")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
