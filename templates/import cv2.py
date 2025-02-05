import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Step 1: Load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            # Assuming folder structure: healthy -> 0, diseased -> 1
            if "diseased" in filename:
                labels.append(1)
            else:
                labels.append(0)
    return images, labels

images, labels = load_images_from_folder("path/to/your/dataset")

# Step 2: Preprocess images
def preprocess_image(img):
    # Implement preprocessing steps such as resizing, denoising, normalization, etc.
    # Example: resizing to a fixed size
    img_resized = cv2.resize(img, (100, 100))
    return img_resized

images_preprocessed = [preprocess_image(img) for img in images]

# Step 3: Feature Extraction
def extract_features(img):
    # Extract features from images using techniques like color histograms, texture analysis, etc.
    # Example: flatten the image
    return img.flatten()

X = np.array([extract_features(img) for img in images_preprocessed])
y = np.array(labels)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a classifier
# Example using Support Vector Machine (SVM)
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 6: Prediction
y_pred = svm_classifier.predict(X_test)

# Step 7: Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Identify the disease based on prediction
def identify_disease(prediction):
    if prediction == 1:
        return "Diseased"
    else:
        return "Healthy"

# Example of using the classifier to predict the disease of a new image
new_image = cv2.imread("path/to/your/new/image.jpg")
new_image_preprocessed = preprocess_image(new_image)
new_image_features = extract_features(new_image_preprocessed)
prediction = svm_classifier.predict([new_image_features])[0]
disease = identify_disease(prediction)
print("The pepper leaf is:", disease)