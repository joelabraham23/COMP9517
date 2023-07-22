import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# def calculate_histogram(image, bins=(8, 8, 8)):
#     # Compute the histogram
#     hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
#     # Normalize the histogram
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist

def calculate_histogram(image, bins=(8, 8, 8)):
    # Define a mask to exclude white pixels
    mask = cv2.inRange(image, np.array([240, 240, 240]), np.array([255, 255, 255]))
    mask = cv2.bitwise_not(mask)  # invert the mask so white pixels are excluded
    # Compute the histogram
    hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 256, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

categories = ["penguin", "turtle"]  # add more if needed
data = []
labels = []

for category in categories:
    for file in glob.glob(f"color/archive/train/train/{category}_*.jpg"):
        image = cv2.imread(file)
        histogram = calculate_histogram(image)
        data.append(histogram)
        labels.append(category)


# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a classifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

# Test the classifier
score = classifier.score(X_test, y_test)
print(f"Classification accuracy: {score}")

# Predictions
predictions = classifier.predict(X_test)
confusion = confusion_matrix(y_test, predictions)
print(confusion)



def predict_image_category(image_path):
    image = cv2.imread(image_path)
    histogram = calculate_histogram(image)
    prediction = classifier.predict([histogram])
    # convert numerical labels back to original labels
    predicted_label = le.inverse_transform(prediction)
    return predicted_label[0]


# print(predict_image_category("archive/valid/valid/image_id_069.jpg", classifier, le))