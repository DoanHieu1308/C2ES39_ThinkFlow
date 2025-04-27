import json
import os
import time
import tempfile
from flask import Blueprint, request, jsonify
from controller.text_to_summary_controller import summarize_and_structure , summarize_to_text_model 
from controller.inference_controller import transcribe_audio_parallel

summary_routes = Blueprint('summary_routes', __name__)

# Mindmap from text input
@summary_routes.route('/mindmap-from-text', methods=['POST'])
def summarize_tree():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({
            "status": 400,
            "message": "Thiếu trường 'text' trong request body."
        }), 400

    result_json = summarize_and_structure(data["text"])

    try:
        result_dict = json.loads(result_json)
    except json.JSONDecodeError:
        return jsonify({
           "status": 500,
           "message": "Không thể phân tích phản hồi từ AI."
        }), 500

    return jsonify(result_dict), result_dict.get("status", 200)


# Đưa ra json mindmap từ audio
@summary_routes.route("/summary_from_audio_to_mindmap", methods=["POST"])
def summary_from_audio_to_mindmap():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = transcribe_audio_parallel(tmp_path)
    finally:
        os.remove(tmp_path)

    # Gọi AI để đưa ra json mindmap
    summary_result_json = summarize_and_structure(text)

    try:
        summary_result_dict = json.loads(summary_result_json)
    except json.JSONDecodeError:
        return jsonify({
           "status": 500,
           "message": "Không thể phân tích phản hồi từ AI."
        }), 500

    elapsed_time = time.time() - start_time
    print(f"⏱️ Thời gian hoàn thành: {elapsed_time:.2f} giây")

    return jsonify(summary_result_dict) , summary_result_dict.get("status", 200)

# Summary from text
@summary_routes.route('/summary-from-text', methods=['POST'])
def summarize_from_text():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({
                "status": 400,
                "message": "Thiếu trường 'text' trong request body."
            }), 400

        result_json = summarize_to_text_model(data["text"])
        return jsonify({"summary": result_json})
    except json.JSONDecodeError:
        return jsonify({
           "status": 500,
           "message": "Không thể phân tích phản hồi từ AI."
        }), 500

# Đưa ra summary từ audio
@summary_routes.route("/summary_from_audio_to_text", methods=["POST"])
def summary_from_audio_to_text():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = transcribe_audio_parallel(tmp_path)
    finally:
        os.remove(tmp_path)

    # Gọi AI để tóm tắt văn bản
    summary_result = summarize_to_text_model(text)

    elapsed_time = time.time() - start_time
    print(f"⏱️ Thời gian hoàn thành: {elapsed_time:.2f} giây")

    return jsonify({"summary_from_audio": summary_result, "time_taken_sec": round(elapsed_time, 2)})

