import face_recognition
import numpy as np
import joblib
from flask import Flask, request, jsonify
import cv2

app = Flask(__name__)

# Load the trained model and label encoder
clf = joblib.load("face_recognition_model.joblib")
le = joblib.load("label_encoder.joblib")  # Load the LabelEncoder properly


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]

    # Log the received file information
    print(
        f"Received file: {image_file.filename}, Content type: {image_file.content_type}"
    )

    if image_file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    if image_file:
        try:
            # Load the image file
            image = face_recognition.load_image_file(image_file)
            print("Image loaded successfully.")
        except Exception as e:
            return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

        # Log image properties
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Convert the image to RGB (face_recognition expects RGB format)
        face_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to a smaller size (optional)
        original_height, original_width, _ = face_image_rgb.shape
        if max(original_height, original_width) > 1024:
            scale_factor = 1024 / max(original_height, original_width)
            new_height = int(original_height * scale_factor)
            new_width = int(original_width * scale_factor)
            face_image_rgb = cv2.resize(face_image_rgb, (new_width, new_height))
            print(f"Resized image to ({new_width}, {new_height})")

        # Detect face locations using the 'cnn' model with increased upsampling
        face_locations = face_recognition.face_locations(
            face_image_rgb, model="cnn", number_of_times_to_upsample=2
        )
        print(f"Detected {len(face_locations)} face locations with CNN model.")

        # If no faces are detected, try the hog model
        if not face_locations:
            face_locations = face_recognition.face_locations(
                face_image_rgb, model="hog", number_of_times_to_upsample=2
            )
            print(f"Detected {len(face_locations)} face locations with HOG model.")

        # Check if any face locations were found
        if not face_locations:
            print("No faces detected.")
            return jsonify({"prediction": "No face detected"}), 200

        # Adjust bounding boxes back to the original image size
        adjusted_face_locations = []
        for top, right, bottom, left in face_locations:
            adjusted_top = int(top / scale_factor)
            adjusted_right = int(right / scale_factor)
            adjusted_bottom = int(bottom / scale_factor)
            adjusted_left = int(left / scale_factor)
            adjusted_face_locations.append(
                (adjusted_top, adjusted_right, adjusted_bottom, adjusted_left)
            )

        # Get face encodings
        face_encodings = face_recognition.face_encodings(face_image_rgb, face_locations)
        print(f"Detected {len(face_encodings)} face encodings.")

        results = []

        if len(face_encodings) == 0:
            print("No face encodings found.")
            return jsonify({"prediction": "No face detected"}), 200

        for face_encoding, (top, right, bottom, left) in zip(
            face_encodings, adjusted_face_locations
        ):
            probabilities = clf.predict_proba([face_encoding])[0]
            max_prob = np.max(probabilities)

            if max_prob < 0.7:  # Adjust this threshold as needed
                predicted_label = "Unknown"
            else:
                predicted_label_index = clf.predict([face_encoding])[0]
                predicted_label = le.inverse_transform([predicted_label_index])[0]

            # Get bounding box coordinates
            results.append(
                {
                    "label": predicted_label,
                    "bounding_box": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left,
                    },
                    "probability": max_prob,
                }
            )

        return jsonify({"predictions": results}), 200
    else:
        return jsonify({"error": "No image provided"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
