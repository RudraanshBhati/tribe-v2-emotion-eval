"""
tribe_inference_emotional.py
TRIBE v2 Inference using emotionally expressive audio (ElevenLabs-generated).

Batched: all 100 audio files are passed to predict() in a single DataLoader
call instead of 100 separate calls. This avoids reinitialising the 20-worker
DataLoader per sample (~2.5 min each) and cuts total runtime from ~4 hours
to ~10-15 minutes.

How batching works:
  - Each audio file is assigned timeline=<sample_id> in the events DataFrame.
  - All per-file DataFrames are concatenated into one combined DataFrame.
  - predict() builds the DataLoader once for all files.
  - Output segments carry segment.timeline = sample_id, so we split preds back
    per sample after inference.

Usage:
  1. Run generate_audio_elevenlabs.py first  ->  emotional_audio/<id>.wav
  2. Run this script                         ->  tribe_activations_emotional.json

Output is directly comparable to tribe_activations.json (same schema).
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

HF_REPO      = "facebook/tribev2"
CACHE_FOLDER = "./tribe_cache_emotional"
AUDIO_DIR    = Path("emotional_audio")
N_VERTICES   = 20_484
OUTPUT_FILE  = "tribe_activations_emotional.json"

ROI_SLICES: dict[str, tuple[int, int]] = {
    "vmPFC":       (0,     2967),
    "amygdala":    (2967,  5721),
    "insula":      (5721,  8476),
    "ACC":         (8476,  11302),
    "TPJ":         (11302, 14478),
    "hippocampus": (14478, 17659),
    "motor":       (17659, 20484),
}


def aggregate_rois(activation: np.ndarray) -> tuple[dict, dict]:
    roi_means, roi_stds = {}, {}
    for name, (s, e) in ROI_SLICES.items():
        region = activation[s:e]
        roi_means[name] = float(region.mean())
        roi_stds[name]  = float(region.std())
    return roi_means, roi_stds


def zero_record(sid: str, text: str) -> dict[str, Any]:
    return {
        "id": sid, "text": text, "n_segments": 0,
        "roi_means": {r: 0.0 for r in ROI_SLICES},
        "roi_stds":  {r: 0.0 for r in ROI_SLICES},
    }


def main() -> None:
    print("=" * 60)
    print("TRIBE v2 Inference -- Emotional Audio (batched)")
    print("=" * 60)

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"{AUDIO_DIR}/ not found. Run generate_audio_elevenlabs.py first."
        )

    with open("mosei_samples.json", encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)
    print(f"Loaded {len(samples)} samples")

    missing = [s["id"] for s in samples if not (AUDIO_DIR / f"{s['id']}.wav").exists()]
    if missing:
        print(f"WARNING: {len(missing)} audio files missing: {missing[:5]}")

    # ── Load TRIBE v2 (once) ──────────────────────────────────────
    print(f"\nLoading TRIBE v2 from {HF_REPO}...")
    from tribev2.demo_utils import TribeModel, get_audio_and_text_events
    model = TribeModel.from_pretrained(HF_REPO, cache_folder=CACHE_FOLDER)
    print("TRIBE v2 loaded.\n")

    # ── Build combined events DataFrame (one row per audio file) ──
    print("Building combined events DataFrame for all audio files...")
    frames: list[pd.DataFrame] = []
    no_audio: list[str] = []

    for sample in samples:
        sid = sample["id"]
        wav = AUDIO_DIR / f"{sid}.wav"
        if not wav.exists():
            no_audio.append(sid)
            continue
        event = {
            "type":     "Audio",
            "filepath": str(wav),
            "start":    0,
            "timeline": sid,       # used to split preds back per sample
            "subject":  "default",
        }
        df = get_audio_and_text_events(pd.DataFrame([event]))
        frames.append(df)

    if not frames:
        print("ERROR: No audio files found.")
        return

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined events: {len(combined)} rows across {len(frames)} audio files\n")

    # ── Single predict() call — DataLoader built once ─────────────
    print("Running TRIBE v2 predict (single DataLoader pass)...")
    preds, all_segments = model.predict(combined, verbose=True)
    print(f"Inference done. Total segments kept: {len(all_segments)}\n")

    # ── Split predictions back by timeline (= sample_id) ─────────
    # preds shape: (n_kept_segments, N_VERTICES)
    # all_segments[i].timeline gives the sample_id for preds[i]
    seg_preds: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, seg in enumerate(all_segments):
        seg_preds[seg.timeline].append(preds[i])

    # ── Assemble results in original sample order ─────────────────
    results: list[dict[str, Any]] = []
    for sample in samples:
        sid  = sample["id"]
        text = sample["text"]

        if sid in no_audio:
            print(f"  [WARN] No audio for {sid} — zero record")
            results.append(zero_record(sid, text))
            continue

        sample_preds = seg_preds.get(sid, [])
        if not sample_preds:
            print(f"  [WARN] No segments predicted for {sid} — zero record")
            results.append(zero_record(sid, text))
            continue

        activation = np.stack(sample_preds).mean(axis=0)
        roi_means, roi_stds = aggregate_rois(activation)
        results.append({
            "id":        sid,
            "text":      text,
            "n_segments": len(sample_preds),
            "roi_means": roi_means,
            "roi_stds":  roi_stds,
        })

    # ── Save ──────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} records -> {OUTPUT_FILE}")

    zero_count = sum(1 for r in results if r["n_segments"] == 0)
    if zero_count:
        print(f"WARNING: {zero_count} samples got zero segments (check audio length/quality)")

    # ── Preview ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROI VALUES -- first 3 samples")
    print("=" * 60)
    for r in results[:3]:
        print(f"\nID: {r['id']} | n_segments: {r['n_segments']}")
        print(f"Text: {r['text'][:80]}")
        print(f"  {'ROI':<14} {'Mean':>10}  {'Std':>10}")
        print("  " + "-" * 36)
        for roi in ROI_SLICES:
            print(f"  {roi:<14} {r['roi_means'][roi]:>10.4f}  {r['roi_stds'][roi]:>10.4f}")


if __name__ == "__main__":
    main()
