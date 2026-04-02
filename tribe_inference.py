"""
tribe_inference.py
TRIBE v2 Inference Script — TRIBE v2 Validation Experiment

Uses META's pretrained TRIBE v2 (facebook/tribev2) to predict brain activations
from text. The model internally converts text to speech (gTTS) and transcribes
it to word-level events before running the multimodal brain encoder.

Pipeline per sample:
  text string
    → write to temp .txt file
    → TribeModel.get_events_dataframe(text_path=...)
        [gTTS TTS + Whisper transcription, cached to ./tribe_cache]
    → TribeModel.predict(events)
        → (n_segments, 20484) fMRI predictions on fsaverage5 cortical mesh
    → mean across segments → (20484,) activation vector
    → aggregate into 7 ROI means + stds

Output: tribe_activations.json
  Fields per sample: id, text, n_segments, roi_means (dict), roi_stds (dict)

Vertex layout (fsaverage5, 10242 vertices per hemisphere):
  Left hemisphere : indices  0     – 10241
  Right hemisphere: indices  10242 – 20483

ROI index ranges (proportionally scaled from standard neuroscience parcellation):
  vmPFC       :  0     – 2967
  amygdala    :  2967  – 5721
  insula      :  5721  – 8476
  ACC         :  8476  – 11302
  TPJ         :  11302 – 14478
  hippocampus :  14478 – 17659
  motor       :  17659 – 20484

NOTE: These are index-based approximations for the validation experiment.
      For publication-grade ROI analysis, replace with atlas-based parcellation
      (e.g. Glasser 2016 HCP-MMP1.0 atlas mapped to fsaverage5).
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────

HF_REPO: str = "facebook/tribev2"
CACHE_FOLDER: str = "./tribe_cache"   # gTTS + transcription cache
N_VERTICES: int = 20_484              # fsaverage5: 2 × 10,242

# ROI index ranges — proportionally scaled to fsaverage5 vertex count
ROI_SLICES: dict[str, tuple[int, int]] = {
    "vmPFC":       (0,     2967),
    "amygdala":    (2967,  5721),
    "insula":      (5721,  8476),
    "ACC":         (8476,  11302),
    "TPJ":         (11302, 14478),
    "hippocampus": (14478, 17659),
    "motor":       (17659, 20484),
}


# ─── ROI aggregation ──────────────────────────────────────────────────────────

def aggregate_rois(
    activation: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Slice a (20484,) activation vector into ROI regions and compute mean + std.

    Args:
        activation: 1-D numpy array of shape (N_VERTICES,)

    Returns:
        roi_means: dict mapping ROI name -> mean activation in that region
        roi_stds:  dict mapping ROI name -> std  activation in that region
    """
    roi_means: dict[str, float] = {}
    roi_stds: dict[str, float] = {}
    for roi_name, (start, end) in ROI_SLICES.items():
        region = activation[start:end]
        roi_means[roi_name] = float(region.mean())
        roi_stds[roi_name]  = float(region.std())
    return roi_means, roi_stds


# ─── Per-sample inference ─────────────────────────────────────────────────────

def infer_single(
    model: Any,
    sample_id: str,
    text: str,
    tmp_dir: Path,
) -> dict[str, Any]:
    """
    Run TRIBE v2 inference on one text sample.

    Writes the text to a temp .txt file, calls model.get_events_dataframe()
    (which runs gTTS + transcription, both cached), then calls model.predict()
    to obtain fMRI predictions. Aggregates across temporal segments by mean.

    Args:
        model:     loaded TribeModel instance
        sample_id: unique sample identifier (for logging)
        text:      raw text string
        tmp_dir:   directory for temporary .txt files

    Returns:
        dict with keys: id, text, n_segments, roi_means, roi_stds
    """
    txt_path = tmp_dir / f"{sample_id}.txt"
    txt_path.write_text(text, encoding="utf-8")

    try:
        events = model.get_events_dataframe(text_path=str(txt_path))
        preds, segments = model.predict(events, verbose=False)
        # preds: (n_segments, 20484)  — each row is one TR-length prediction

        if preds.shape[0] == 0:
            # No segments returned (very short text after transcription)
            # Fall back to zeros so the pipeline doesn't break
            activation = np.zeros(N_VERTICES, dtype=np.float32)
            n_segments = 0
        else:
            activation = preds.mean(axis=0)   # mean across TRs → (20484,)
            n_segments = preds.shape[0]

    finally:
        txt_path.unlink(missing_ok=True)

    roi_means, roi_stds = aggregate_rois(activation)

    return {
        "id":         sample_id,
        "text":       text,
        "n_segments": n_segments,
        "roi_means":  roi_means,
        "roi_stds":   roi_stds,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    """Load TRIBE v2, run inference on all 100 samples, save tribe_activations.json."""
    print("=" * 60)
    print("TRIBE v2 Inference — Real Model (facebook/tribev2)")
    print("=" * 60)

    # ── Load samples ──────────────────────────────────────────────────────────
    input_path = "mosei_samples.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Run load_mosei.py first."
        )
    with open(input_path, encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)
    print(f"\nLoaded {len(samples)} samples from {input_path}")

    # ── Load TRIBE v2 ─────────────────────────────────────────────────────────
    print(f"\nLoading TRIBE v2 from HuggingFace ({HF_REPO})...")
    print(f"Cache folder: {CACHE_FOLDER}")
    print("(First run downloads ~10 GB of model weights — subsequent runs use cache)\n")

    from tribev2.demo_utils import TribeModel

    model = TribeModel.from_pretrained(HF_REPO, cache_folder=CACHE_FOLDER)
    print("TRIBE v2 loaded.\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    results: list[dict[str, Any]] = []
    failed: list[str] = []
    tmp_dir = Path("tribe_tmp")
    tmp_dir.mkdir(exist_ok=True)

    print(f"Running inference on {len(samples)} samples...")
    print("Note: gTTS + transcription results are cached in ./tribe_cache")
    print("      First run is slow; cached samples are near-instant.\n")

    for sample in tqdm(samples, desc="TRIBE v2 inference", unit="sample"):
        sid  = sample["id"]
        text = sample["text"]
        try:
            result = infer_single(model, sid, text, tmp_dir)
            results.append(result)
        except Exception as exc:
            print(f"\n  [WARN] Failed on {sid}: {exc}")
            failed.append(sid)
            # Insert zeros so downstream scripts still have 100 entries
            roi_means = {roi: 0.0 for roi in ROI_SLICES}
            roi_stds  = {roi: 0.0 for roi in ROI_SLICES}
            results.append({
                "id": sid, "text": text,
                "n_segments": 0,
                "roi_means": roi_means,
                "roi_stds":  roi_stds,
            })

    # Clean up temp dir
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = "tribe_activations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} activation records -> {output_path}")

    if failed:
        print(f"Failed samples ({len(failed)}): {failed}")

    # ── Print first 3 samples ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROI VALUES — first 3 samples")
    print("=" * 60)
    for r in results[:3]:
        print(f"\nID         : {r['id']}")
        print(f"Text       : {r['text'][:100]}")
        print(f"N segments : {r['n_segments']}")
        print(f"{'ROI':<14} {'Mean':>12}  {'Std':>12}")
        print("-" * 42)
        for roi in ROI_SLICES:
            m = r["roi_means"][roi]
            s = r["roi_stds"][roi]
            print(f"  {roi:<12} {m:>12.4f}  {s:>12.4f}")


if __name__ == "__main__":
    main()
