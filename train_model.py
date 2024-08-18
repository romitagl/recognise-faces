import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
X = np.load("face_features.npy")
y = np.load("face_labels.npy")
le_classes = np.load("label_encoder.npy", allow_pickle=True)

# Recreate the LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.classes_ = le_classes

print("X shape:", X.shape)
print("y shape:", y.shape)

# Check Class Distribution
unique, counts = np.unique(y, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution:", class_distribution)


# Stratified Train-Test Split
if len(X) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    print("Not enough samples to split. Using all data for training.")
    X_train, y_train = X, y
    X_test, y_test = X, y


# Train the classifier model
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# Save the model
joblib.dump(clf, "face_recognition_model.joblib")

# Make predictions
y_pred = clf.predict(X_test)

# Print classification report with original labels
# Use zero_division Parameter
# When printing the classification report, you can set the zero_division parameter to control how to handle cases where a label has no true samples. Setting it to 0 will replace undefined metrics with 0.
print(
    classification_report(
        le.inverse_transform(y_test), le.inverse_transform(y_pred), zero_division=0
    )
)

# Save the LabelEncoder with the model for future use
joblib.dump(le, "label_encoder.joblib")


### Output analysis
# python train_model.py
# X shape: (53, 128)
# y shape: (53,)
# Class distribution: {0: 17, 1: 19, 2: 17}
#                 precision    recall  f1-score   support

#      Person_A       1.00      0.75      0.86         4
#      Person_B       0.75      0.75      0.75         4
#      Unknown       0.75      1.00      0.86         3

#     accuracy                           0.82        11
#    macro avg       0.83      0.83      0.82        11
# weighted avg       0.84      0.82      0.82        11

# Analysis of the Classification Report
# Precision, Recall, and F1-Score:
# Person_A: Precision is 1.00, indicating that all predictions for this class were correct, but the recall is 0.75, meaning that 25% of Person_A's instances were missed.
# Person_B: Precision and recall are both 0.75, indicating a balanced performance but showing room for improvement.
# Unknown: This class has a high recall (1.00), meaning all instances of "Unknown" were correctly identified, but precision is lower (0.75), indicating some misclassifications.
# Accuracy: The overall accuracy of 0.82 indicates that the model is correctly predicting 82% of the instances.
# Macro and Weighted Averages: The macro average shows balanced performance across classes, while the weighted average accounts for class imbalance, reflecting the model's performance more accurately given the distribution of classes.
