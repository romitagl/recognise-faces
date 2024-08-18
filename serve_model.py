import numpy as np
import joblib
from flask import Flask, request, jsonify
import cv2
import dlib

app = Flask(__name__)

# Load the trained model and label encoder
# clf: a trained classifier model
# le: the LabelEncoder object
clf = joblib.load("face_recognition_model.joblib")
le = joblib.load("label_encoder.joblib")

# Load dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)


def align_face(image, face):
    shape = shape_predictor(image, face)
    face_chip = dlib.get_face_chip(image, shape)
    return face_chip


def extract_face_encoding(face_chip):
    face_descriptor = face_rec_model.compute_face_descriptor(face_chip)
    return np.array(face_descriptor)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if image_file:
        try:
            # Read the image file
            image_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

        # Detect faces using dlib
        faces = face_detector(
            image_rgb, 1
        )  # 1 is the number of times to upsample the image

        if not faces:
            return jsonify({"prediction": "No face detected"}), 200

        results = []

        for face in faces:
            # Align and extract face encoding
            face_chip = align_face(image_rgb, face)
            face_encoding = extract_face_encoding(face_chip)

            # Predict
            probabilities = clf.predict_proba([face_encoding])[0]
            max_prob = np.max(probabilities)

            if max_prob < 0.7:  # Adjust this threshold as needed
                predicted_label = "Unknown"
            else:
                predicted_label_index = clf.predict([face_encoding])[0]
                predicted_label = le.inverse_transform([predicted_label_index])[0]

            # Get bounding box coordinates
            left, top, right, bottom = (
                face.left(),
                face.top(),
                face.right(),
                face.bottom(),
            )
            results.append(
                {
                    "label": predicted_label,
                    "bounding_box": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left,
                    },
                    "probability": float(max_prob),
                }
            )

        return jsonify({"predictions": results}), 200
    else:
        return jsonify({"error": "No image provided"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
