import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM extraction only")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid extraction")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    mode = "hybrid" if args.hybrid else "llm" if args.use_llm else "rules"
    record_id = audio_path.stem

    raw_transcript = transcribe_audio(str(audio_path), model_size=args.whisper_model)
    text = PREPROCESS(raw_transcript)

    rec = build_base_record(record_id=record_id, mode=mode, model=args.model)

    if args.hybrid:
        apply_hybrid(text, rec, args.model)
    elif args.use_llm:
        apply_llm(text, rec, args.model)
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

    print(f"Audio processed successfully.")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()