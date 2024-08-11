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
    print(f"Received file: {image_file.filename}, Content type: {image_file.content_type}")

    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if image_file:  # Check if image_file is not None
        # Check if the file is an image
        if image_file.content_type.startswith("image/"):
            try:
                image = face_recognition.load_image_file(image_file)
            except Exception as e:
                return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

            face_encodings = face_recognition.face_encodings(image)
            face_locations = face_recognition.face_locations(image)

            results = []

            if len(face_encodings) == 0:
                return jsonify({"prediction": "No face detected"}), 200

            for face_encoding, face_location in zip(face_encodings, face_locations):
                probabilities = clf.predict_proba([face_encoding])[0]
                max_prob = np.max(probabilities)

                if max_prob < 0.7:  # Adjust this threshold as needed
                    predicted_label = "Unknown"
                else:
                    predicted_label_index = clf.predict([face_encoding])[0]
                    predicted_label = le.inverse_transform([predicted_label_index])[0]

                # Get bounding box coordinates
                top, right, bottom, left = face_location
                results.append({
                    "label": predicted_label,
                    "bounding_box": {
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "left": left
                    },
                    "probability": max_prob
                })

            # Optionally, draw bounding boxes on the image
            image_with_boxes = draw_boxes_on_image(image, results)

            # Convert the image with boxes to a byte array for response
            _, img_encoded = cv2.imencode('.jpg', image_with_boxes)
            img_bytes = img_encoded.tobytes()

            return jsonify({"predictions": results, "image_with_boxes": img_bytes.hex()}), 200
        else:
            return jsonify({"error": "File is not an image"}), 400
    else:
        return jsonify({"error": "No image provided"}), 400

def draw_boxes_on_image(image, results):
    """Draw bounding boxes and labels on the image."""
    # Convert the image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for result in results:
        label = result["label"]
        bounding_box = result["bounding_box"]
        top, right, bottom, left = bounding_box["top"], bounding_box["right"], bounding_box["bottom"], bounding_box["left"]

        # Draw the bounding box
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        # Put the label text
        cv2.putText(image_bgr, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_bgr

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
