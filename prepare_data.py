import os
import json
import face_recognition
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image, ImageDraw, ExifTags
import logging
import cv2
import dlib
import multiprocessing
from functools import partial


def extract_and_encode_face(face_image, face_location):
    """
    Extract and encode the face from the given image using the provided location.
    """
    # Convert the face_location (top, right, bottom, left) to a dlib rectangle
    top, right, bottom, left = face_location
    face_rect = dlib.rectangle(left, top, right, bottom)

    # Load the dlib shape predictor and face recognition model
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1(
        "dlib_face_recognition_resnet_model_v1.dat"
    )

    try:
        # Detect facial landmarks in the face image
        shape = shape_predictor(face_image, face_rect)

        # Compute the face descriptor (128D face encoding)
        face_encoding = face_rec_model.compute_face_descriptor(face_image, shape)

        # Convert the dlib.vector to a NumPy array
        face_encoding_np = np.array(face_encoding)

        return face_encoding_np

    except Exception as e:
        print(f"Error in encoding face: {e}")
        return None


def polygon_to_bbox(polygon):
    x_coordinates, y_coordinates = zip(*polygon)
    return [
        int(min(x_coordinates)),
        int(min(y_coordinates)),
        int(max(x_coordinates)),
        int(max(y_coordinates)),
    ]


def rotate_image(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image


def process_image(json_file, images_dir, annotations_dir, output_dir):
    X = []
    y = []

    if not json_file.endswith(".json"):
        return X, y

    json_path = os.path.join(annotations_dir, json_file)
    img_file = os.path.splitext(json_file)[0] + ".jpg"
    img_path = os.path.join(images_dir, img_file)

    if not os.path.exists(img_path):
        logging.warning(f"No image file found for {json_file}")
        return X, y

    logging.info(f"Processing {img_file}")

    # Load and rotate image
    with Image.open(img_path) as img:
        img = rotate_image(img)
        image = np.array(img)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    logging.info(f"Rotated image shape: {image.shape}")

    # Load annotations
    with open(json_path, "r") as f:
        annotation = json.load(f)

    # Process each face in the image
    for shape in annotation["shapes"]:
        label = shape["label"]
        polygon = shape["points"]

        # Convert polygon to bounding box
        bbox = polygon_to_bbox(polygon)
        top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
        face_image = image[top:bottom, left:right]

        # Draw original bounding box in blue
        draw.rectangle([(left, top), (right, bottom)], outline="blue", width=2)

        # Ensure face image is in the correct format
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        try:
            # Detect face locations in the cropped face image
            face_locations = face_recognition.face_locations(
                face_image_rgb, model="cnn"
            )

            # If no faces are detected, try the hog model
            if not face_locations:
                face_locations = face_recognition.face_locations(
                    face_image_rgb, model="hog"
                )

            if face_locations:
                # Process each detected face
                for f_top, f_right, f_bottom, f_left in face_locations:
                    # Adjust face location to original image coordinates
                    f_top += top
                    f_bottom += top
                    f_left += left
                    f_right += left

                    # Draw detected bounding box in green
                    draw.rectangle(
                        [(f_left, f_top), (f_right, f_bottom)], outline="green", width=2
                    )

                    # Extract face encoding
                    encoding = extract_and_encode_face(
                        image, (f_top, f_right, f_bottom, f_left)
                    )
                    if encoding is not None:
                        X.append(encoding)
                        y.append(label)
                    else:
                        logging.warning(
                            f"No face encoding found for {img_file}, label: {label}"
                        )

            else:
                logging.warning(f"No faces detected in {img_file} for label {label}")

        except Exception as e:
            logging.error(
                f"Unexpected error processing {img_file}, label: {label}. Error: {str(e)}"
            )

    # Save the annotated image
    output_path = os.path.join(output_dir, f"annotated_{img_file}")
    pil_image.save(output_path)
    logging.info(f"Saved annotated image to {output_path}")

    return X, y


def prepare_data(data_dir, output_dir):
    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations")

    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(annotations_dir) if f.endswith(".json")]

    # Create a partial function with fixed arguments
    process_image_partial = partial(
        process_image,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
    )

    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count() - 1  # Leave one core free
    num_cores = max(1, num_cores)  # Ensure at least one core is used

    logging.info(f"Using {num_cores} CPU cores")

    # Use Pool for multiprocessing
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_image_partial, json_files)

    # Combine results
    all_X = []
    all_y = []
    for X, y in results:
        all_X.extend(X)
        all_y.extend(y)

    if not all_X:
        logging.warning(
            "No face encodings were generated. Check your input data and face detection parameters."
        )
        return None, None, None

    X = np.array(all_X)
    le = LabelEncoder()
    y = le.fit_transform(all_y)

    print(f"Processed {len(X)} faces")
    logging.info(f"Total labels found: {len(y)}, Unique labels: {np.unique(y)}")

    if len(np.unique(y)) < 2:
        logging.warning(
            "Only one class detected. Ensure your dataset has multiple classes."
        )

    return X, y, le


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Usage
    data_dir = "./dataset"
    output_dir = "./output"
    X, y, le = prepare_data(data_dir, output_dir)

    if X is not None:
        # Save the prepared data
        np.save("face_features.npy", X)
        np.save("face_labels.npy", y)
        np.save("label_encoder.npy", le.classes_)

        logging.info(f"Processed {len(X)} faces")
        logging.info(f"Labels: {le.classes_}")
    else:
        logging.warning(
            "No face encodings were generated. Check the logs for more information."
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
