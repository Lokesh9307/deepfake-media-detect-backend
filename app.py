from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from image_model import predict_image
from video_model import predict_video
from audio_model import predict_audio
import os
from dotenv import load_dotenv
load_dotenv()

PORT= os.getenv("PORT", 5000)

app = Flask(__name__)
CORS(app)  # Apply CORS correctly

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    media_type = request.form.get('type')  # image, video, audio
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        if media_type == "image":
            label, chart = predict_image(path)
        elif media_type == "video":
            label, chart = predict_video(path)
        elif media_type == "audio":
            label, chart = predict_audio(path)
        else:
            return jsonify({"error": "Invalid media type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"label": label, "chart": chart})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Server is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

