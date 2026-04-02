# TRIBE v2 Validation Experiment -- Project Status

**Last updated:** 2026-04-02
**Working directory:** `d:/Projects/tribe-eval`
**Environment:** Windows 11, RTX 4070 (12GB VRAM), Python 3.12.4, uv 0.10.11, CUDA 12.1

---

## What This Project Does

Validates whether META's TRIBE v2 brain encoding model can help LLMs understand emotions better.

**Pipeline:**
```
Emotional Audio (ElevenLabs TTS) -> TRIBE v2 (META's pretrained model)
  -> predicted brain activations (20,484 vertices)
  -> brain_formatter.py (converts to natural language)
  -> "Neural Emotion Context"
  -> llama3.1:8b (via Ollama, local) + original text -> emotion prediction
```

**3 experimental conditions:**
- **A** -- Text only (baseline)
- **B** -- Text + real TRIBE v2 neural context
- **C** -- Text + shuffled neural context (control -- mismatched brain data)

**Validation logic:** If B > A > C on implicit/masked emotions -> neural context carries real signal worth building a full system around.

---

## BOTH PIPELINE RUNS COMPLETE

### Run 1 -- Text-based (tribe_activations.json)
Results in `results_text.json` / `results_summary_text.csv`

| Category | A | B | C | Signal? |
|----------|---|---|---|---------|
| Overall  | 52% | 54% | 52% | PARTIAL |
| Explicit | 60% | 62.5% | 55% | B>A>C |
| Implicit | 50% | 52.5% | 55% | NOISE (C>B) |
| Sarcastic| 40% | 40% | 40% | Flat |

### Run 2 -- Emotional Audio (tribe_activations_emotional.json)
Results in `results_emotional.json` / `results_summary_emotional.csv`

| Category | A | B | C | Signal? |
|----------|---|---|---|---------|
| Overall  | 52% | 55% | 54% | PARTIAL |
| Explicit | 60% | **67.5%** | 60% | B>A=C (stronger) |
| Implicit | 50% | 50% | 55% | NOISE (C>B) |
| Sarcastic| 40% | 40% | 40% | Flat |

**Key delta:** Emotional audio improved B on explicit by +5pp (62.5% -> 67.5%).
VAD-Valence correlation for B dropped (0.629 -> 0.595).

**Verdict: PARTIAL SIGNAL** (both runs). Decision gate saved -> `decision_gate.txt`

---

## Completed Steps

### Environment Setup (done)
- `uv init tribe-eval` -- all deps managed via pyproject.toml
- torch 2.5.1+cu121 installed via explicit PyTorch index
- Ollama at `C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe` (not in PATH for bash -- Python library works)
- `llama3.1:8b` pulled and verified
- `tribev2` cloned from `github.com/facebookresearch/tribev2`, editable workspace member
- `elevenlabs` 2.41.0 installed via pip (not in pyproject.toml -- free-tier TTS tool)
- **pyproject.toml note:** `requires-python = ">=3.12,<3.14"`, numpy pinned `>=2.2.6`

### Step 1 -- Data Loader (done) (`load_mosei.py`)
- Uses `dair-ai/emotion` from HuggingFace (CMU-MOSEI server dead)
- 100 samples: 40 explicit, 40 implicit, 20 sarcastic
- Output: `mosei_samples.json`

### Step 2 -- Emotional Audio Generation (done) (`generate_audio_elevenlabs.py`)
- ElevenLabs API, free tier (9,307 / 10,000 chars used -- likely exhausted now)
- 12 voices, 3 per emotion, rotated round-robin: happy->Laura/Jessica/Charlie, sad->George/Brian/Sarah, angry->Harry/Adam/Callum, fearful->Liam/Lily/River
- Voice settings tuned per emotion (stability/style) in addition to voice identity
- Output: `emotional_audio/<id>.wav` (16kHz mono, 100 files)
- API key in `.env` (gitignored). Mark Leslie excluded -- `professional` category, not free via API.
- Run: `python generate_audio_elevenlabs.py`

### Step 3a -- TRIBE v2 Text-based Inference (done) (`tribe_inference.py`)
- Uses cached text->word events (Whisper not needed -- text already in mosei_samples.json)
- ~14s/sample x 100 = ~23 min total
- Output: `tribe_activations.json` (100 records, real TRIBE v2 weights, values ~0.03-0.07)

### Step 3b -- TRIBE v2 Emotional Audio Inference (done) (`tribe_inference_emotional.py`)
- Batched: all 100 WAVs -> one combined events DataFrame -> single predict() call
- Combined events: 2,197 rows across 100 audio files
- 580 segments kept from 10,000 (5.8%)
- ~10-15 min total (vs ~4 hours per-sample approach)
- Output: `tribe_activations_emotional.json`
- **VRAM note:** TRIBE v2 uses ~8-10GB. Cannot run simultaneously with Ollama.

### Step 4 -- Brain-to-Text Formatter (done x2)
- Text run: reads `tribe_activations.json` -> `formatted_contexts_text.json`
- Emotional run: reads `tribe_activations_emotional.json` -> `formatted_contexts_emotional.json`
- Current state of `brain_formatter.py`: reads emotional, outputs emotional
- Z-scores each ROI, maps to qualitative levels, generates `[Neural Emotion Context]` blocks
- Also generates shuffled context (derangement) for Condition C

### Step 5 -- LLM Evaluation (done x2) (`evaluate.py`)
- Text run: reads `formatted_contexts_text.json` (but currently script reads emotional -- see note)
- Emotional run: reads `formatted_contexts_emotional.json`, outputs `results_emotional.json`
- **NOTE:** `evaluate.py` currently configured for emotional run (reads formatted_contexts_emotional.json, suffix=_emotional)
- 3 conditions x 100 samples = 300 Ollama calls to llama3.1:8b
- Results: `results.json` (text-based), `results_emotional.json` (emotional)
- Summary: `results_summary.csv` (text-based), `results_summary_emotional.csv` (emotional)

### Step 6 -- Visualizer (done) (`visualize_results.py`)
- Last run was on emotional results (plots/ reflect emotional audio run)
- Reads `results_summary.csv` by default -- swap to emotional by copying before run
- Outputs: `plots/accuracy_by_category.png`, `plots/vad_correlation_overall.png`,
  `plots/vad_valence_heatmap.png`, `decision_gate.txt`

---

## What To Do Next

### Option A -- Direct ROI Classifier (recommended next step)
Write a script that runs logistic regression directly on the 7 ROI means to predict emotion label.
Bypasses the LLM entirely. Shows how much discriminative signal exists in the raw activations.
Compare text-based vs emotional audio activations.

### Option B -- Glasser HCP-MMP1.0 Atlas Parcellation
Replace index-based ROI slices with real anatomical parcellation (360 regions).
Requires re-running TRIBE v2 inference with atlas-based slicing.
More meaningful neuroscientific interpretation.

### Option C -- Emotion Activation Fingerprint Heatmap
4 emotions x 7 ROIs heatmap of mean z-scored activations.
Shows whether different emotions produce distinct activation patterns.
Could be the core figure for any publication.

---

## File Inventory

| File | Status | Description |
|------|--------|-------------|
| `pyproject.toml` | done | Single source of truth for deps |
| `load_mosei.py` | done | Data loader |
| `mosei_samples.json` | done | 100 samples (40 explicit, 40 implicit, 20 sarcastic) |
| `generate_audio_elevenlabs.py` | done | ElevenLabs TTS, 12 voices across 4 emotions |
| `emotional_audio/` | done | 100 WAVs at 16kHz mono |
| `tribe_inference.py` | done | Text-based TRIBE v2 inference |
| `tribe_inference_emotional.py` | done | Batched TRIBE v2 inference on emotional audio |
| `tribe_activations.json` | done | Text-based TRIBE v2 brain activations |
| `tribe_activations_emotional.json` | done | Emotional audio TRIBE v2 brain activations |
| `brain_formatter.py` | done* | *Currently reads emotional; converts activations -> NL context |
| `formatted_contexts_text.json` | done | Text-based neural contexts (Conditions B+C) |
| `formatted_contexts_emotional.json` | done | Emotional audio neural contexts (Conditions B+C) |
| `evaluate.py` | done* | *Currently configured for emotional run |
| `results.json` | done | Text-based evaluation results (raw) |
| `results_text.json` | done | Text-based results backup |
| `results_emotional.json` | done | Emotional audio evaluation results (raw) |
| `results_summary.csv` | done | Text-based metrics summary |
| `results_summary_text.csv` | done | Text-based metrics summary backup |
| `results_summary_emotional.csv` | done | Emotional audio metrics summary |
| `visualize_results.py` | done | Charts + decision gate |
| `plots/` | done | PNG charts (from emotional run) |
| `decision_gate.txt` | done | Final verdict (PARTIAL SIGNAL, from emotional run) |

---

## Key Technical Notes

- **Activate venv before every command:** `cd d:/Projects/tribe-eval && source .venv/Scripts/activate`
- **Ollama not in PATH:** Use Python `ollama` library directly
- **No Unicode in print statements:** Windows cp1252 encoding -- use ASCII only
- **uv only for deps:** `uv add` or `uv sync` -- except `elevenlabs` (installed via pip --frozen workaround)
- **All inference local:** TRIBE v2 on GPU, Ollama on GPU -- NOT simultaneously (12GB VRAM limit)
- **ElevenLabs .env:** API key in `tribe-eval/.env` (gitignored), auto-loaded by generate script
- **Batched inference:** `tribe_inference_emotional.py` builds one DataLoader for all 100 files. Uses `timeline=sid` to split predictions back per sample via `segment.timeline`.
- **ROI slices** (proportionally scaled to 20,484 vertices):
  ```
  vmPFC       :     0 -  2967
  amygdala    :  2967 -  5721
  insula      :  5721 -  8476
  ACC         :  8476 - 11302
  TPJ         : 11302 - 14478
  hippocampus : 14478 - 17659
  motor       : 17659 - 20484
  ```
- **For publication:** Replace index-based ROI slicing with Glasser HCP-MMP1.0 atlas parcellation
- **z-score clustering issue:** Many samples cluster in "moderate" (z in -0.5 to 0.5). This limits formatter expressiveness. Real atlas parcellation would likely spread variance more.
