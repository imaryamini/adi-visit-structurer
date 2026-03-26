from pathlib import Path
from flask import Flask, render_template, request, jsonify

from src.voice_input import transcribe_audio
from src.run_pipeline import (
    PREPROCESS,
    build_base_record,
    apply_rules,
    postprocess_record,
    run_quality_check,
)

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/assistant")
def assistant():
    return render_template("index.html")


@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = UPLOAD_DIR / audio_file.filename
    audio_file.save(save_path)

    raw_transcript = transcribe_audio(str(save_path))
    text = PREPROCESS(raw_transcript)

    rec = build_base_record(record_id="web_demo", mode="rules", model="")
    apply_rules(text, rec)
    postprocess_record(rec, text)

    q = run_quality_check(rec, text)
    rec["quality"]["missing_mandatory_fields"] = q.get(
        "missing_mandatory_fields",
        q.get("missing_fields", []),
    )
    rec["quality"]["warnings"] = q.get("warnings", [])

    return jsonify({
        "transcript": raw_transcript,
        "result": rec
    })


@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("text") or "").strip()

    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    text = PREPROCESS(raw_text)

    rec = build_base_record(record_id="web_text_demo", mode="rules", model="")
    apply_rules(text, rec)
    postprocess_record(rec, text)

    q = run_quality_check(rec, text)
    rec["quality"]["missing_mandatory_fields"] = q.get(
        "missing_mandatory_fields",
        q.get("missing_fields", []),
    )
    rec["quality"]["warnings"] = q.get("warnings", [])

    return jsonify({
        "transcript": raw_text,
        "result": rec
    })


if __name__ == "__main__":
    app.run(debug=True)