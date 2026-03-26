import argparse
import json
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from src.run_pipeline import (
    PREPROCESS,
    PIPELINE_VERSION,
    build_base_record,
    apply_rules,
    apply_llm,
    apply_hybrid,
    postprocess_record,
    run_quality_check,
)
from src.voice_input import transcribe_audio


OUTPUT_DIR = Path("reports/audio_demo")
RECORDINGS_DIR = Path("data/audio_recordings")


def record_audio(output_path: Path, duration: int, samplerate: int = 16000) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    sf.write(str(output_path), audio, samplerate)
    print(f"Saved recording to: {output_path}")
    return output_path


def process_audio_file(audio_path: Path, use_llm: bool, hybrid: bool, model: str, whisper_model: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "hybrid" if hybrid else "llm" if use_llm else "rules"
    record_id = audio_path.stem

    raw_transcript = transcribe_audio(str(audio_path), model_size=whisper_model)
    text = PREPROCESS(raw_transcript)

    rec = build_base_record(record_id=record_id, mode=mode, model=model)

    if hybrid:
        apply_hybrid(text, rec, model)
    elif use_llm:
        apply_llm(text, rec, model)
    else:
        apply_rules(text, rec)

    postprocess_record(rec, text)

    q = run_quality_check(rec, text)
    rec["quality"]["missing_mandatory_fields"] = q.get(
        "missing_mandatory_fields",
        q.get("missing_fields", []),
    )
    rec["quality"]["warnings"] = q.get("warnings", [])

    payload = {
        "record_id": record_id,
        "audio_file": str(audio_path),
        "transcript_raw": raw_transcript,
        "transcript_processed": text,
        "prediction": rec,
        "pipeline_version": PIPELINE_VERSION,
    }

    out_path = OUTPUT_DIR / f"{record_id}_audio_result.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTranscription:")
    print(raw_transcript)
    print(f"\nStructured output saved to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=15, help="Recording duration in seconds")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM extraction only")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid extraction")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = RECORDINGS_DIR / f"visit_note_{timestamp}.wav"

    record_audio(audio_path, duration=args.duration)
    process_audio_file(
        audio_path=audio_path,
        use_llm=args.use_llm,
        hybrid=args.hybrid,
        model=args.model,
        whisper_model=args.whisper_model,
    )


if __name__ == "__main__":
    main()