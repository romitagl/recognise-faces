# Description

Python program to recognise faces.
Initial images are not labelled, therefore it is necessary to manually label them.
A machine learning model is trained using the custom dataset to recognise Person A or Person B. In case the person is not recognised, the model will return "Unknown".

## Steps

1. Dataset Preparation and Labeling
2. Environment Setup
3. Face Detection and Feature Extraction
4. Model Training
5. Model Evaluation
6. Dockerization
7. Serving the Model

### Step 1: Dataset Preparation and Labeling

Image folder structure should look like this:

```text
dataset/
│
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
│
└── annotations/
    ├── image1.json
    ├── image2.json
    ├── image3.json
    └── ...
```

In this structure:

All images are stored in a single images/ folder.
Each image has a corresponding JSON file in the annotations/ folder.

The JSON files will contain the face locations and labels for each person in the image.

For each face, we have a label (Person_A, Person_B, or Unknown) and a bbox (bounding box) containing [x, y, width, height] coordinates.

Here is an example json:

```json
{
    "version": "5.5.0",
    "flags": {},
    "shapes": [
        {
            "label": "Person_A",
            "points": [
                [
                    3084.421052631579,
                    309.36842105263173
                ],
                [
                    3979.1578947368425,
                    335.68421052631595
                ],
                [
                    3994.9473684210525,
                    1619.8947368421054
                ],
                [
                    3031.7894736842104,
                    1646.2105263157896
                ]
            ],
            "group_id": null,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": null
        },
        {
            "label": "Unknown",
            "points": [
                [
                    2326.5263157894738,
                    377.7894736842107
                ],
                [
                    2963.3684210526317,
                    383.0526315789475
                ],
                [
                    2873.8947368421054,
                    1219.8947368421054
                ],
                [
                    2316.0,
                    1209.3684210526317
                ]
            ],
            "group_id": null,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": null
        },
        {
            "label": "Person_B",
            "points": [
                [
                    884.421052631579,
                    440.9473684210528
                ],
                [
                    1537.0526315789475,
                    519.8947368421054
                ],
                [
                    1494.9473684210525,
                    1077.7894736842106
                ],
                [
                    837.0526315789475,
                    1067.2631578947369
                ]
            ],
            "group_id": null,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": null
        }
    ],
    "imageHeight": 3024,
    "imageWidth": 4032
}
```

To manually label images, a good option is: `labelme`: An open-source annotation tool that can export to JSON format.

To install and use labelme: `pip install labelme`,  and then to use labelme: `labelme`.

Open your image directory and annotate each face with a rectangle, labeling it as "Person_A", "Person_B", or "Unknown".
Save the annotations. labelme will create a JSON file for each image.

You can label multiple faces in a single image.
You can handle "Unknown" faces explicitly.
The annotation format is flexible and can be easily parsed in Python.

### Step 2: Environment Setup

Create a new directory for your project.
Set up a virtual environment:

```bash
python3 -m venv face_recognition_env
source face_recognition_env/bin/activate
```

Install required libraries: `pip install --no-cache-dir -r requirements.txt`

### Step 3: Face Detection and Feature Extraction

Create a Python script (e.g., prepare_data.py) to detect faces and extract features.

This script does the following:

It expects a data_dir with images and annotations subfolders.
It iterates through each image in the images folder.
For each image, it loads the corresponding JSON annotation file.
It processes each face in the annotation:

Extracts the face region using the bounding box coordinates.
Computes the face encoding for the extracted face.
Adds the face encoding to X and the label to y.

It uses a LabelEncoder to convert string labels to numeric labels.
Finally, it saves the prepared data (face features, labels, and label encoder) as NumPy files.

To use this script: Make sure your dataset is organized as described, with images and annotations subfolders.
Update the data_dir variable with the path to your dataset folder.

Make sure that the file dlib_face_recognition_resnet_model_v1.dat is available in your working directory or provide the correct path to it.
You can download the model file from Dlib's official model zoo: `curl -o dlib_face_recognition_resnet_model_v1.dat.bz2 http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2`
And for the shape predictor: `curl -o shape_predictor_68_face_landmarks.dat.bz2 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2`

After downloading, extract the .bz2 file to get the .dat file and place it in the correct directory: `bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2`, `bzip2 -d shape_predictor_68_face_landmarks.dat.bz2`

Run the script: `python prepare_data.py`

The process is as follows:

1. Load the image and rotate it based on its EXIF data.
2. Convert the rotated image to a numpy array for processing.
3. Draw the original bounding boxes from the annotations on the rotated image.
4. Attempt to detect faces within these bounding boxes.
5. If faces are detected, draw the detected face bounding boxes in green.
6. Save the annotated image with both the original (blue) and detected (green) bounding boxes.

### Step 4: Model Training

Use the script (e.g., train_model.py) to train a simple classifier.

### Step 5: Model Evaluation

The classification report printed at the end of the training script will give you an idea of the model's performance. You can also create a separate evaluation script if needed.

### Step 6: Dockerization

Use the Dockerfile in your project directory.

### Step 7: Serving the Model

Use the script (e.g., serve_model.py) to serve your model.

Now you can submit new images to <http://localhost:5555/predict> for prediction. Example using ./img1.jpg: `curl -X POST -F "image=@img1.jpg" http://localhost:5555/predict`.

Note: The -F option allows you to send form data, which is what Flask expects for file uploads.

You can also use the test_model.py script to test your model: `python test_model.py path/to/your/image.jpg`