# src/voice_input.py

from pathlib import Path
import whisper

# ---------------------------
# CONFIG
# ---------------------------

# OPTIONS:
# "base"  -> fast (recommended)
# "small" -> more accurate (slower)
MODEL_NAME = "base"

# ---------------------------
# LOAD MODEL ONCE
# ---------------------------

try:
    model = whisper.load_model(MODEL_NAME)
    print(f"[Whisper] Model loaded: {MODEL_NAME}")
except Exception as e:
    raise RuntimeError(f"Failed to load Whisper model: {e}")


# ---------------------------
# TRANSCRIPTION FUNCTION
# ---------------------------

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Whisper.

    Features:
    - Automatic language detection (Italian + English)
    - Optimized for clinical dictation
    - Fast + stable for demo
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        result = model.transcribe(
            str(audio_path),
            task="transcribe",  # NO translation
            initial_prompt=(
                "Trascrizione di una visita medica domiciliare ADI. "
                "Include parametri vitali come pressione, frequenza cardiaca, saturazione, temperatura."
            ),
            fp16=False,  # safe for CPU (Mac)
            verbose=False
        )

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        # Debug log (keep for demo, can remove later)
        print(f"[Whisper] Language detected: {language}")
        print(f"[Whisper] Transcript: {text}")

        if not text:
            return "No speech detected."

        return text

    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")