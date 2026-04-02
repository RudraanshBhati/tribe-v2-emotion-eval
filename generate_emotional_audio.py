"""
generate_emotional_audio.py
Generates emotionally expressive speech for each sample using edge-tts
(Microsoft AriaNeural voice) with prosody adjustments per emotion.

Edge-tts does NOT support mstts:express-as (Azure-only feature), but
AriaNeural with prosody tuning (pitch/rate/volume) gives distinct acoustic
profiles per emotion that TRIBE v2's audio encoder will respond to differently
than flat gTTS audio.

Emotion -> prosody mapping:
  happy    : rate=+25%, pitch=+8Hz, volume=+10%   (fast, high, loud)
  sad      : rate=-30%, pitch=-8Hz, volume=-10%   (slow, low, quiet)
  fearful  : rate=+20%, pitch=+12Hz               (fast, high — panic)
  angry    : rate=+15%, pitch=+8Hz, volume=+20%   (loud, higher)

Even without style tags, AriaNeural is a neural voice significantly more
expressive than gTTS's robotic monotone. Combined with prosody tuning this
gives TRIBE v2 genuinely varied audio signal to work with.

Output: emotional_audio/<sample_id>.wav (16kHz mono, via ffmpeg)
Speed:  ~0.5-1s per sample (cloud TTS, cached locally)
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

INPUT_FILE   = "mosei_samples.json"
OUTPUT_DIR   = Path("emotional_audio")
VOICE        = "en-US-AriaNeural"

# Prosody per emotion — rate=%, pitch=Hz, volume=%
# These give TRIBE v2's audio encoder distinct acoustic features per emotion.
EMOTION_PROSODY: dict[str, dict[str, str]] = {
    "happy":   {"rate": "+25%", "pitch": "+8Hz",  "volume": "+10%"},
    "sad":     {"rate": "-30%", "pitch": "-8Hz",  "volume": "-10%"},
    "fearful": {"rate": "+20%", "pitch": "+12Hz", "volume": "+0%"},
    "angry":   {"rate": "+15%", "pitch": "+8Hz",  "volume": "+20%"},
    # synonyms
    "joy":     {"rate": "+25%", "pitch": "+8Hz",  "volume": "+10%"},
    "sadness": {"rate": "-30%", "pitch": "-8Hz",  "volume": "-10%"},
    "fear":    {"rate": "+20%", "pitch": "+12Hz", "volume": "+0%"},
    "anger":   {"rate": "+15%", "pitch": "+8Hz",  "volume": "+20%"},
    "neutral": {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
}

FFMPEG = r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"


def mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    result = subprocess.run(
        [FFMPEG, "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")


async def generate_one(sid: str, text: str, label: str,
                        mp3_path: Path, wav_path: Path) -> float:
    """Generate audio file. Returns duration in seconds."""
    import edge_tts

    prosody = EMOTION_PROSODY.get(label.lower(), EMOTION_PROSODY["neutral"])
    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=prosody["rate"],
        pitch=prosody["pitch"],
        volume=prosody["volume"],
    )
    await communicate.save(str(mp3_path))
    mp3_to_wav(mp3_path, wav_path)
    mp3_path.unlink(missing_ok=True)
    size = wav_path.stat().st_size
    return size / (16000 * 2)   # seconds at 16kHz 16-bit mono


async def main() -> None:
    import edge_tts  # noqa

    print("=" * 60)
    print("Emotional Audio Generation -- edge-tts AriaNeural + prosody")
    print(f"Voice: {VOICE}")
    print("=" * 60)
    print("\nProsody mapping:")
    for label, p in EMOTION_PROSODY.items():
        if label in ("happy", "sad", "fearful", "angry"):
            print(f"  {label:<10} rate={p['rate']:<6} pitch={p['pitch']:<8} volume={p['volume']}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        samples: list[dict] = json.load(f)
    print(f"\nLoaded {len(samples)} samples")

    OUTPUT_DIR.mkdir(exist_ok=True)
    tmp_dir = OUTPUT_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    generated = skipped = failed = 0

    for i, sample in enumerate(samples):
        sid   = sample["id"]
        text  = sample["text"]
        label = sample.get("emotion_label", "neutral")

        wav_path = OUTPUT_DIR / f"{sid}.wav"
        mp3_path = tmp_dir / f"{sid}.mp3"

        if wav_path.exists():
            skipped += 1
            continue

        prosody = EMOTION_PROSODY.get(label.lower(), EMOTION_PROSODY["neutral"])
        print(f"[{i+1:3d}/100] {sid:<22} {label:<10} rate={prosody['rate']:<6} | {text[:45]}")

        try:
            dur = await generate_one(sid, text, label, mp3_path, wav_path)
            print(f"         {dur:.1f}s")
            generated += 1
        except Exception as exc:
            print(f"  [WARN] Failed: {exc}")
            failed += 1

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nDone. Generated: {generated} | Skipped: {skipped} | Failed: {failed}")
    print(f"Files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    asyncio.run(main())
