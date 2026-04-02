"""
generate_audio_elevenlabs.py
Generates emotionally expressive speech using ElevenLabs API.
Output: emotional_audio/<sample_id>.wav (16kHz mono PCM)

Setup:
    uv add elevenlabs
    Set ELEVENLABS_API_KEY env var, or paste key below.

Credit estimate: ~9,307 chars total -> fits within 10,000 free-tier credits.

Voice assignment (all free-tier pre-made voices, rotated per emotion):
    happy   -> Rachel (F, calm-bright), Domi (F, strong), Charlie (M, casual)
    sad     -> Sarah (F, soft), Thomas (M, calm), Sam (M, raspy)
    angry   -> Clyde (M, war veteran), Arnold (M, crisp), Patrick (M, confident)
    fearful -> Harry (M, anxious), Fin (M, sailor), Dorothy (F, pleasant)
"""

import json
import os
import wave
from pathlib import Path

INPUT_FILE = "mosei_samples.json"
OUTPUT_DIR = Path("emotional_audio")

# Load API key from .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.startswith("ELEVENLABS_API_KEY="):
            os.environ.setdefault("ELEVENLABS_API_KEY", _line.split("=", 1)[1].strip())

API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_API_KEY_HERE")

MODEL_ID = "eleven_turbo_v2"   # cheapest model, counts fewest chars vs quality tradeoff

SAMPLE_RATE = 16000  # Hz — must match TRIBE v2 audio encoder expectation

# Voices confirmed available on this account (all premade/professional).
# 3 per emotion, rotated round-robin, chosen to match emotional character.
EMOTION_VOICES: dict[str, list[str]] = {
    "happy": [
        "FGY2WhTYpPnrIDTdsKH5",  # Laura   — Enthusiast, Quirky
        "cgSgspJ2msm6clMCkdW9",  # Jessica — Playful, Bright, Warm
        "IKne3meq5aSn9XLyUdCD",  # Charlie — Energetic, Confident
    ],
    "sad": [
        "JBFqnCBsd6RMkjVDRZzb",  # George      — Warm, Captivating
        "nPczCjzI2devNBz1zQrb",  # Brian       — Deep, Resonant, Comforting
        "EXAVITQu4vr4xnSDxMaL",  # Sarah       — Mature, Reassuring
    ],
    "angry": [
        "SOYHLrjzK2X1ezoPC6cr",  # Harry  — Fierce Warrior
        "pNInz6obpgDQGcFmaJgB",  # Adam   — Dominant, Firm
        "N2lVS1w4EtoT3dr4eOWO",  # Callum — Husky, Edgy
    ],
    "fearful": [
        "TX3LPaxmHKxFdv7VOQHJ",  # Liam    — Energetic (nervous energy)
        "pFZP5JQG7iQjIQuC4Bku",  # Lily    — Velvety, Soft
        "SAz9YHcvj6GT2YYXdXww",  # River   — Neutral, Subdued
    ],
    "neutral": [
        "CwhRBWXzGAHq8TQ4Fs17",  # Roger
    ],
}
EMOTION_VOICES["joy"]     = EMOTION_VOICES["happy"]
EMOTION_VOICES["sadness"] = EMOTION_VOICES["sad"]
EMOTION_VOICES["fear"]    = EMOTION_VOICES["fearful"]
EMOTION_VOICES["anger"]   = EMOTION_VOICES["angry"]

# Voice settings per emotion
EMOTION_SETTINGS: dict[str, dict] = {
    "happy":   {"stability": 0.30, "similarity_boost": 0.80, "style": 0.70, "use_speaker_boost": True},
    "sad":     {"stability": 0.70, "similarity_boost": 0.75, "style": 0.25, "use_speaker_boost": False},
    "fearful": {"stability": 0.35, "similarity_boost": 0.80, "style": 0.60, "use_speaker_boost": True},
    "angry":   {"stability": 0.20, "similarity_boost": 0.90, "style": 0.90, "use_speaker_boost": True},
    "neutral": {"stability": 0.50, "similarity_boost": 0.75, "style": 0.00, "use_speaker_boost": False},
}
EMOTION_SETTINGS["joy"]     = EMOTION_SETTINGS["happy"]
EMOTION_SETTINGS["sadness"] = EMOTION_SETTINGS["sad"]
EMOTION_SETTINGS["fear"]    = EMOTION_SETTINGS["fearful"]
EMOTION_SETTINGS["anger"]   = EMOTION_SETTINGS["angry"]


def pcm_to_wav(pcm_bytes: bytes, wav_path: Path) -> None:
    """Wrap raw 16-bit PCM bytes in a WAV container."""
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def generate_one(client, text: str, label: str, voice_id: str, wav_path: Path) -> float:
    """Generate one WAV file. Returns audio duration in seconds."""
    from elevenlabs import VoiceSettings

    s = EMOTION_SETTINGS.get(label.lower(), EMOTION_SETTINGS["neutral"])
    settings = VoiceSettings(
        stability=s["stability"],
        similarity_boost=s["similarity_boost"],
        style=s["style"],
        use_speaker_boost=s["use_speaker_boost"],
    )

    # pcm_16000 = raw signed 16-bit little-endian PCM at 16kHz mono
    audio_bytes = b"".join(
        client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=MODEL_ID,
            voice_settings=settings,
            output_format="pcm_16000",
        )
    )

    pcm_to_wav(audio_bytes, wav_path)
    return len(audio_bytes) / (SAMPLE_RATE * 2)  # seconds


def main() -> None:
    from elevenlabs import ElevenLabs

    if API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Set ELEVENLABS_API_KEY env var or paste your key into API_KEY above.")
        return

    client = ElevenLabs(api_key=API_KEY)

    with open(INPUT_FILE, encoding="utf-8") as f:
        samples: list[dict] = json.load(f)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Count samples per emotion for rotation tracking
    emotion_counts: dict[str, int] = {}

    total_chars = sum(len(s["text"]) for s in samples)
    voice_names = {
        "FGY2WhTYpPnrIDTdsKH5": "Laura",
        "cgSgspJ2msm6clMCkdW9": "Jessica",
        "IKne3meq5aSn9XLyUdCD": "Charlie",
        "JBFqnCBsd6RMkjVDRZzb": "George",
        "nPczCjzI2devNBz1zQrb": "Brian",
        "EXAVITQu4vr4xnSDxMaL": "Sarah",
        "SOYHLrjzK2X1ezoPC6cr": "Harry",
        "pNInz6obpgDQGcFmaJgB": "Adam",
        "N2lVS1w4EtoT3dr4eOWO": "Callum",
        "TX3LPaxmHKxFdv7VOQHJ": "Liam",
        "pFZP5JQG7iQjIQuC4Bku": "Lily",
        "SAz9YHcvj6GT2YYXdXww": "River",
        "CwhRBWXzGAHq8TQ4Fs17": "Roger",
    }

    print("=" * 60)
    print(f"ElevenLabs Audio Generation -- {len(samples)} samples")
    print(f"Model: {MODEL_ID}  |  Total chars: {total_chars} / 10000 free")
    print("=" * 60)
    print()
    for em in ("happy", "sad", "angry", "fearful"):
        names = ", ".join(voice_names.get(v, v) for v in EMOTION_VOICES[em])
        print(f"  {em:<10} -> {names}")
    print()

    generated = skipped = failed = 0

    for i, sample in enumerate(samples):
        sid   = sample["id"]
        text  = sample["text"]
        label = sample.get("emotion_label", "neutral")
        wav_path = OUTPUT_DIR / f"{sid}.wav"

        if wav_path.exists():
            skipped += 1
            continue

        # Round-robin voice selection per emotion
        idx = emotion_counts.get(label, 0)
        voices = EMOTION_VOICES.get(label.lower(), EMOTION_VOICES["neutral"])
        voice_id = voices[idx % len(voices)]
        emotion_counts[label] = idx + 1

        vname = voice_names.get(voice_id, voice_id)
        print(f"[{i+1:3d}/100] {sid:<22} {label:<10} {vname:<12} | {text[:40]}")

        try:
            dur = generate_one(client, text, label, voice_id, wav_path)
            print(f"         {dur:.1f}s -> {wav_path.name}")
            generated += 1
        except Exception as exc:
            print(f"  [WARN] Failed: {exc}")
            failed += 1

    print(f"\nDone. Generated: {generated} | Skipped: {skipped} | Failed: {failed}")
    print(f"Files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
