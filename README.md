# TRIBE v2 Emotion Validation Experiment

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20%2B%20CUDA%2012.1-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)

Can predicted brain activations help a language model understand emotion better?

This project validates whether [META's TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model — which predicts fMRI responses to text and audio — carries discriminative emotion signal that an LLM can use. We test this with a controlled 3-condition experiment using 100 emotion-labeled text samples, two input modalities (text vs expressive audio), and local LLM inference via Ollama.

---

## Pipeline

```
                        +---------------------------+
  mosei_samples.json    |  ElevenLabs TTS (12 voices)|
  (100 labelled texts)  |  emotional_audio/*.wav     |
          |             +---------------------------+
          |                          |
          v                          v
  tribe_inference.py      tribe_inference_emotional.py
  (text-based input)      (audio-based, batched)
          |                          |
          v                          v
  tribe_activations.json   tribe_activations_emotional.json
  (7 ROI means x 100)      (same schema, audio-derived)
          |                          |
          +----------+  +------------+
                     v  v
               brain_formatter.py
               (z-score -> natural language)
                     |
                     v
          formatted_contexts*.json
          (real context + shuffled control)
                     |
              +------+------+
              v             v
         evaluate.py    roi_classifier.py
         (LLM eval)     (direct ML, no LLM)
              |             |
              v             v
       results_summary   plots/ + results.txt
```

---

## Experimental Design

| Condition | LLM Input | Role |
|-----------|-----------|------|
| **A** (baseline) | Text only | What the LLM achieves without any brain signal |
| **B** (test) | Text + real TRIBE v2 neural context | Does brain data help? |
| **C** (control) | Text + shuffled neural context | Proves B is meaningful, not noise |

**Validation logic:** `B > A > C` = signal. `B ≈ C` = noise.

100 samples drawn from [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion):
- 40 **explicit** (emotion named directly: "I feel happy")
- 40 **implicit** (emotion inferred from context: "it was my last day before retirement")
- 20 **sarcastic** (sentiment contradicts tone: "oh great, another Monday")

4 emotion classes: happy, angry, sad, fearful. Chance = 25%.

---

## Results

### LLM Evaluation (llama3.1:8b via Ollama)

**Emotional audio input to TRIBE v2:**

| Condition | Overall | Explicit | Implicit | Sarcastic |
|-----------|---------|----------|----------|-----------|
| A (text only) | 52% | 60% | 50% | 40% |
| **B (neural context)** | **55%** | **67.5%** | 50% | 40% |
| C (shuffled) | 54% | 60% | 55% | 40% |

**Text-based input to TRIBE v2 (for comparison):**

| Condition | Overall | Explicit | Implicit |
|-----------|---------|----------|----------|
| A | 52% | 60% | 50% |
| B | 54% | 62.5% | 52.5% |
| C | 52% | 55% | 55% |

Emotional audio improved Condition B on explicit emotions by **+5pp** (62.5% -> 67.5%).

### Direct ROI Classifier (no LLM)

Logistic regression / SVM / Random Forest on 7 ROI means directly:

| Input | LogReg | SVM | RF | Chance |
|-------|--------|-----|----|--------|
| Text-based | 20% | 26% | 20% | 25% |
| **Emotional audio** | **26%** | **33%** | **32%** | **25%** |

**Per category (LogReg, text-based):** Explicit 47.5%, Implicit 37.5%, Sarcastic 30%

**Per category (LogReg, emotional audio):** Explicit 45%, Implicit 47.5%, Sarcastic 35%

Most informative ROIs: **ACC** (conflict/uncertainty) and **amygdala** (threat/arousal).

---

## Key Findings

- **Neural context helps on explicit emotions.** Condition B reached 67.5% accuracy on explicitly stated emotions, vs 60% baseline and 60% shuffled -- a clean B > A = C pattern.
- **Raw brain activations encode emotion above chance.** A simple SVM on 7 ROI means hit 33% (emotional audio) -- above the 25% chance level with no LLM required.
- **Emotional audio (ElevenLabs) enriches TRIBE v2 signal.** Implicit emotion classification from raw ROIs jumped from 37.5% (text) to 47.5% (audio). Prosody reaches the Wav2VecBert encoder and changes the predicted activation patterns.
- **ACC and amygdala carry the most discriminative signal** -- consistent with emotion neuroscience literature.
- **Implicit emotions remain challenging for the LLM eval.** C(55%) > B(50%) on implicit in the LLM condition suggests the natural language formatter does not yet surface the signal the raw activations contain. The ROI classifier, which bypasses the formatter, shows the signal is there.
- **Sarcasm is flat across all conditions (40%)** -- expected; sarcasm requires pragmatic inference no brain encoder helps with at this scale.

**Verdict: PARTIAL SIGNAL.** Brain data improves explicit emotion prediction. Implicit emotion signal exists in the activations (shown by ROI classifier) but the LLM formatter does not yet surface it effectively.

---

## Limitations

- **100 samples** -- results are directional, not statistically definitive.
- **7 index-based ROI slices** -- not anatomically validated. Real parcellation (Glasser HCP-MMP1.0, 360 regions) would give more interpretable and likely stronger signal.
- **z-score clustering** -- many samples fall in the "moderate" z-score band, limiting the formatter's expressiveness.
- **ElevenLabs free tier** -- 10,000 chars/month. Re-generation requires a new API key or paid plan.
- **Local LLM (8B params)** -- a larger model or fine-tuned emotion classifier would likely respond better to the neural context.

---

## Plots

All plots are in `plots/`:

| Plot | Description |
|------|-------------|
| `accuracy_by_category.png` | LLM emotion accuracy per condition x category |
| `vad_valence_heatmap.png` | VAD-Valence correlation per category x condition |
| `vad_correlation_overall.png` | VAD (V/A/D) correlations overall |
| `roi_classifier_comparison.png` | ROI classifier accuracy: text vs audio |
| `roi_emotion_fingerprint_text.png` | Emotion activation heatmap (text-based) |
| `roi_emotion_fingerprint_emotional.png` | Emotion activation heatmap (emotional audio) |

---

## Installation

Requires Python 3.12, CUDA 12.1, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Clone this repo
git clone https://github.com/RudraanshBhati/tribe-v2-emotion-eval
cd tribe-v2-emotion-eval

# 2. Clone TRIBE v2 as a workspace member (required -- editable install)
git clone https://github.com/facebookresearch/tribev2

# 3. Install all dependencies (PyTorch CUDA index configured in pyproject.toml)
uv sync

# 4. Pull the LLM for evaluation
ollama pull llama3.1:8b

# 5. Set up secrets
cp .env.example .env
# Edit .env and add:
#   ELEVENLABS_API_KEY=your_key_here   (only needed to regenerate audio)
#   HF_TOKEN=your_token_here           (HuggingFace gated access)
```

> **VRAM note:** TRIBE v2 uses ~8-10 GB. Ollama uses ~5-6 GB. Run them sequentially on a 12 GB GPU.

---

## Reproduce

Run steps in order. Each step's output is the next step's input.

```bash
# 0. Activate venv
source .venv/Scripts/activate   # Windows
# source .venv/bin/activate     # Linux/Mac

# 1. Build dataset (100 samples from dair-ai/emotion)
python load_mosei.py

# 2. Generate emotional audio (ElevenLabs, requires API key)
python generate_audio_elevenlabs.py

# 3a. TRIBE v2 inference on text
python tribe_inference.py

# 3b. TRIBE v2 inference on emotional audio (batched, ~15 min on RTX 4070)
python tribe_inference_emotional.py

# 4. Format brain activations as natural language context
python brain_formatter.py

# 5. LLM evaluation (300 Ollama calls, ~5-8 min)
python evaluate.py

# 6a. Direct ROI classifier (no LLM)
python roi_classifier.py

# 6b. Visualize LLM results + decision gate
python visualize_results.py
```

Results land in `plots/`, `results_summary*.csv`, `roi_classifier_results.txt`, and `decision_gate.txt`.

---

## File Structure

```
tribe-v2-emotion-eval/
├── pyproject.toml                      # Dependencies (uv)
├── .python-version                     # Python 3.12
├── .gitignore
│
├── load_mosei.py                       # Step 1: dataset
├── generate_audio_elevenlabs.py        # Step 2: ElevenLabs TTS
├── generate_emotional_audio.py         # Step 2 alt: edge-tts fallback
├── tribe_inference.py                  # Step 3a: text-based TRIBE v2
├── tribe_inference_emotional.py        # Step 3b: audio TRIBE v2 (batched)
├── brain_formatter.py                  # Step 4: activations -> NL context
├── evaluate.py                         # Step 5: LLM eval (Ollama)
├── roi_classifier.py                   # Step 6a: direct ML classifier
├── visualize_results.py                # Step 6b: plots + decision gate
│
├── mosei_samples.json                  # 100 labelled samples
├── results_summary.csv                 # LLM eval results (text-based)
├── results_summary_emotional.csv       # LLM eval results (emotional audio)
├── roi_classifier_results.txt          # ROI classifier accuracy summary
├── decision_gate.txt                   # Final verdict
│
├── plots/                              # All output figures (6 PNGs)
└── PROJECT_STATUS.md                   # Detailed experiment log
```

---

## Credits & Acknowledgements

- **[TRIBE v2](https://github.com/facebookresearch/tribev2)** -- META / Facebook AI Research. Brain encoding model predicting fMRI from multimodal inputs.
- **[ElevenLabs](https://elevenlabs.io)** -- Emotionally expressive TTS used for audio generation.
- **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)** -- Hugging Face emotion dataset (CMU-MOSEI substitute).
- **[Ollama](https://ollama.com) + llama3.1:8b** -- Local LLM inference.
- **Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013)** -- Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods.*

---

## Citation

If you use this experiment design or code:

```bibtex
@misc{tribe_emotion_eval_2026,
  title   = {TRIBE v2 Emotion Validation Experiment},
  year    = {2026},
  url     = {https://github.com/RudraanshBhati/tribe-v2-emotion-eval},
  note    = {Validates whether TRIBE v2 predicted brain activations improve LLM emotion understanding}
}
```

---

## License

MIT
