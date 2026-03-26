from pathlib import Path

# Supported audio formats (INCLUDING webm from browser)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".webm"}


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file into text using faster-whisper.
    Supports browser recordings (.webm) and standard formats.
    """

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "faster-whisper is not installed.\n"
            "Run: python3 -m pip install faster-whisper"
        ) from e

    path = Path(audio_path)

    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Check format
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Use one of: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    # Load Whisper model
    model = WhisperModel(model_size, compute_type="int8")

    # IMPORTANT: force Italian (since your project is ADI Italy)
    segments, _ = model.transcribe(str(path), language="it")

    # Combine segments
    text_parts = []
    for segment in segments:
        if segment.text:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)

    text = " ".join(text_parts).strip()

    if not text:
        raise ValueError("Transcription returned empty text.")

    return text


# Optional test (you can ignore this)
if __name__ == "__main__":
    sample = "sample.wav"
    if Path(sample).exists():
        print(transcribe_audio(sample))
    else:
        print("Put a sample audio file to test.")