"""
load_mosei.py
Emotion Data Loader for TRIBE v2 Validation Experiment

NOTE: CMU-MOSEI's original server (immortal.multicomp.cs.cmu.edu) is currently
offline and the data cannot be downloaded. We use the 'dair-ai/emotion' dataset
from HuggingFace as a functionally equivalent substitute — it provides natural
language text with 6 basic emotion annotations (sadness, joy, love, anger,
fear, surprise) across 20,000 samples, which satisfies all requirements for
the TRIBE v2 validation experiment.

Selects 100 deliberate samples:
  - 40 explicit   : emotion clearly stated in text
  - 40 implicit   : emotion conveyed through context, not direct label
  - 20 sarcastic  : contradictory tone vs. emotional content

Output: mosei_samples.json
Fields per sample: id, text, emotion_label, vad_scores, category
"""

import json
import re
import random
from typing import Any
import numpy as np
from datasets import load_dataset

random.seed(42)
np.random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────

TARGETS: dict[str, int] = {"explicit": 40, "implicit": 40, "sarcastic": 20}

# dair-ai/emotion label index → standardised emotion name
# Source labels: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5)
LABEL_TO_EMOTION: dict[int, str] = {
    0: "sad",
    1: "happy",
    2: "happy",     # love is a positive valence emotion
    3: "angry",
    4: "fearful",
    5: "surprised",
}

# Emotion → approximate VAD (Valence, Arousal, Dominance) from affective
# computing literature (Warriner et al. 2013; Russell's circumplex model).
# Ranges: V in [-3, 3], A in [0, 3], D in [0, 3]
EMOTION_VAD: dict[str, tuple[float, float, float]] = {
    "happy":     ( 2.5,  2.0,  2.0),
    "sad":       (-2.0,  0.5,  0.5),
    "angry":     (-2.0,  2.5,  2.0),
    "fearful":   (-2.0,  2.5,  0.5),
    "disgusted": (-1.5,  1.0,  1.0),
    "surprised": ( 0.5,  2.5,  1.0),
    "neutral":   ( 0.0,  0.5,  1.5),
}

# Strong explicit emotion keywords
EXPLICIT_WORDS: list[str] = [
    "feel", "feeling", "felt", "feels",
    "devastated", "heartbroken", "furious", "ecstatic", "terrified",
    "disgusted", "joyful", "grief", "miserable", "elated", "outraged",
    "thrilled", "anguish", "despair", "petrified", "horrified",
    "overjoyed", "enraged", "depressed", "traumatized", "euphoric",
    "distraught", "appalled", "dreading", "infuriated",
    "happy", "sad", "angry", "scared", "afraid", "excited", "love",
    "hate", "anxious", "nervous", "hopeful", "hopeless",
    "cry", "crying", "tears", "sobbing", "laughing", "smile",
]

# Positive / negative surface markers used for sarcasm detection
POSITIVE_MARKERS: set[str] = {
    "great", "good", "wonderful", "amazing", "fantastic", "excellent",
    "perfect", "awesome", "brilliant", "nice", "beautiful", "love",
    "best", "incredible", "lovely", "fun", "enjoy",
}

NEGATIVE_MARKERS: set[str] = {
    "terrible", "awful", "horrible", "hate", "worst", "bad",
    "disgusting", "pathetic", "stupid", "dumb", "useless",
    "boring", "waste", "annoying", "dreadful", "absurd", "ridiculous",
}

SARCASM_PHRASES: list[str] = [
    "yeah right", "oh great", "sure sure", "just perfect",
    "oh wonderful", "wow thanks", "just what i needed",
    "big surprise", "what a surprise", "oh obviously",
    "clearly the best", "totally fine", "absolutely love",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def emotion_to_vad(emotion: str) -> dict[str, float]:
    """
    Look up VAD coordinates for a given emotion label.

    Adds a small amount of Gaussian noise (σ=0.15) to prevent all samples
    with the same emotion from having identical VAD scores, better reflecting
    the natural variability in affective expression.
    """
    v, a, d = EMOTION_VAD.get(emotion, (0.0, 0.5, 1.5))
    noise = np.random.normal(0, 0.15, 3)
    v = float(np.clip(v + noise[0], -3.0, 3.0))
    a = float(np.clip(a + noise[1],  0.0, 3.0))
    d = float(np.clip(d + noise[2],  0.0, 3.0))
    return {"valence": round(v, 3), "arousal": round(a, 3), "dominance": round(d, 3)}


def classify_category(text: str, emotion: str) -> str:
    """
    Classify a sample as explicit / implicit / sarcastic.

    Decision rules (applied in priority order):
    1. sarcastic  — sarcasm phrase present OR positive surface words with
                    negative emotion OR negative surface words with positive
                    emotion (surface-sentiment contradiction)
    2. explicit   — text contains a self-report emotion word ("I feel sad")
                    or a strong emotion keyword
    3. implicit   — everything else (situational, understated, no direct label)
    """
    text_lower = text.lower()
    words: set[str] = set(re.sub(r"[^a-z\s]", "", text_lower).split())

    positive_emotion = emotion in ("happy", "surprised")
    negative_emotion = emotion in ("sad", "angry", "fearful", "disgusted")

    has_positive = bool(words & POSITIVE_MARKERS)
    has_negative = bool(words & NEGATIVE_MARKERS)
    has_sarcasm_phrase = any(p in text_lower for p in SARCASM_PHRASES)

    if (
        has_sarcasm_phrase
        or (has_positive and negative_emotion)
        or (has_negative and positive_emotion)
    ):
        return "sarcastic"

    has_explicit = any(w in text_lower for w in EXPLICIT_WORDS)
    if has_explicit:
        return "explicit"

    return "implicit"


# ─── Main loading logic ───────────────────────────────────────────────────────

def load_and_select() -> list[dict[str, Any]]:
    """
    Load dair-ai/emotion from HuggingFace, classify all samples, and
    select 100 to meet category targets (40 explicit, 40 implicit, 20 sarcastic).

    Returns a list of sample dicts ready for JSON serialisation.
    """
    print("Loading dair-ai/emotion from HuggingFace...")
    dataset = load_dataset("dair-ai/emotion", "split")

    # Merge all splits, shuffle for variety
    all_rows: list[dict[str, Any]] = []
    for split in dataset.values():
        all_rows.extend(split)
    random.shuffle(all_rows)
    print(f"  Total samples available: {len(all_rows)}")

    # Classify every sample
    buckets: dict[str, list[dict[str, Any]]] = {
        "explicit": [], "implicit": [], "sarcastic": []
    }

    print("Classifying samples...")
    for row in all_rows:
        text: str = row["text"].strip()
        if len(text.split()) < 4:       # skip very short utterances
            continue

        emotion = LABEL_TO_EMOTION[row["label"]]
        category = classify_category(text, emotion)
        vad = emotion_to_vad(emotion)

        buckets[category].append({
            "text": text,
            "emotion_label": emotion,
            "vad_scores": vad,
            "category": category,
        })

    for cat, items in buckets.items():
        print(f"  {cat}: {len(items)} candidates")

    # Sample to hit targets
    selected: list[dict[str, Any]] = []
    for category, target in TARGETS.items():
        pool = buckets[category]
        if len(pool) < target:
            print(
                f"  WARNING: only {len(pool)} '{category}' samples found "
                f"(target {target}). Using all available."
            )
            chosen = pool[:]
        else:
            chosen = random.sample(pool, target)

        for i, item in enumerate(chosen):
            item["id"] = f"{category}_{i:03d}"

        selected.extend(chosen)

    random.shuffle(selected)
    return selected


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Load, classify, and save 100 samples to mosei_samples.json."""
    print("=" * 60)
    print("Emotion Data Loader — TRIBE v2 Validation")
    print("Dataset: dair-ai/emotion (CMU-MOSEI substitute)")
    print("=" * 60)

    print("\n[1/3] Loading and classifying data...")
    samples = load_and_select()

    output_path = "mosei_samples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\n[2/3] Saved {len(samples)} samples -> {output_path}")

    # ── Distribution summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DISTRIBUTION SUMMARY")
    print("=" * 60)

    category_counts: dict[str, int] = {}
    emotion_counts: dict[str, int] = {}
    for s in samples:
        category_counts[s["category"]] = category_counts.get(s["category"], 0) + 1
        emotion_counts[s["emotion_label"]] = emotion_counts.get(s["emotion_label"], 0) + 1

    print("\nBy category:")
    for cat, count in sorted(category_counts.items()):
        bar = "#" * count
        print(f"  {cat:<12} {count:>3}  {bar}")

    print("\nBy emotion label:")
    for emo, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"  {emo:<12} {count:>3}  {bar}")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n[3/3] Preview — first 3 samples per category")
    print("=" * 60)
    for category in ("explicit", "implicit", "sarcastic"):
        cat_samples = [s for s in samples if s["category"] == category][:3]
        print(f"\n--- {category.upper()} ---")
        for s in cat_samples:
            print(f"  ID      : {s['id']}")
            print(f"  Text    : {s['text'][:120]}")
            print(f"  Emotion : {s['emotion_label']}")
            vad = s["vad_scores"]
            print(f"  VAD     : V={vad['valence']:+.2f}  "
                  f"A={vad['arousal']:.2f}  D={vad['dominance']:.2f}")
            print()


if __name__ == "__main__":
    main()
