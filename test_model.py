import requests
import cv2
import os
import sys
import argparse

def overlay_predictions_on_image(image_path):
    # Create a 'prediction' folder if it doesn't exist
    prediction_folder = "predictions"
    os.makedirs(prediction_folder, exist_ok=True)

    # Get the base name of the input file
    base_name = os.path.basename(image_path)
    
    # Create the output path
    output_path = os.path.join(prediction_folder, base_name)

    # Open the image file
    with open(image_path, 'rb') as img_file:
        # Send a POST request to the Flask server
        response = requests.post("http://localhost:5555/predict", files={"image": img_file})

    if response.status_code == 200:
        data = response.json()
        # Check if predictions are available
        if "predictions" in data:
            # Load the original image for overlaying
            original_image = cv2.imread(image_path)
            for prediction in data["predictions"]:
                label = prediction["label"]
                bounding_box = prediction["bounding_box"]
                top = bounding_box["top"]
                right = bounding_box["right"]
                bottom = bounding_box["bottom"]
                left = bounding_box["left"]

                # Draw the bounding box with increased thickness
                cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 8)

                # Increase the font scale for larger labels
                font_scale = 4.8
                # Get the size of the label text
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

                # Adjust the label position to fit above the bounding box
                label_x = left
                label_y = top - 10 if top - 10 - label_height > 0 else top + baseline

                # Draw the label with increased font size and thickness
                cv2.putText(original_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

            # Save the output image with bounding boxes
            cv2.imwrite(output_path, original_image)
            print(f"Output image saved to {output_path}")
        else:
            print("No predictions returned in the response.")
    else:
        print(f"Error: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Process an image and overlay predictions.")
    parser.add_argument("image_path", help="Path to the input image file")
    args = parser.parse_args()

    overlay_predictions_on_image(args.image_path)

if __name__ == "__main__":
    main()
