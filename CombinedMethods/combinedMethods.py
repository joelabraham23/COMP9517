import json
import sys
import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk

from tkinter import filedialog
from PIL import ImageTk, Image
from tkinter import filedialog

sys.path.insert(0, "../Yolov8")
sys.path.insert(1, "../cannySiftNN")
sys.path.insert(2, "../color")
sys.path.insert(3, "../FeatureClass")
sys.path.insert(4, "../haarCascade")

from YoloClassify import main as yolo_main
from cannySiftNN_train import main as canny_sift_main
from densenet import DenseNet
from color_model import predict_image_category
from fClassMachine import main as fClassMain
from main import haarCascade as haar_main

def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select image to classify")
    return file_path

def process_image(image_path):
    yolo_label, yolo_score, yolo_box = yolo_main(image_path)
    canny_sift_label, canny_sift_score = canny_sift_main(image_path)
    color_label, color_proba = predict_image_category(image_path)
    fClass_label, fClass_proba = fClassMain(image_path)
    haar_label, harr_proba, haars_box = haar_main(image_path)

    return {
        "yolo": {
            "label": yolo_label,
            "score": yolo_score,
            "box": yolo_box,
        },
        "cannySiftNN": {
            "label": canny_sift_label,
            "score": canny_sift_score,
        },
        "color": {
            "label": color_label,
            "score": color_proba,
        },
        "Feature Classifier": {
            "label": fClass_label,
            "score": fClass_proba,
        },
        "Haars Cascade": {
            "label": haar_label,
            "score": harr_proba,
            "box": haars_box,
        },
    }


def weighted_result(result_json):
    weights = {"yolo": 0.45, "cannySiftNN": 0.15, "color": 0.15, "Feature Classifier": 0.15, "Haars Cascade":0.10}
    scores = {"penguin": 0.0, "turtle": 0.0}
    for method in result_json.keys():
        label = result_json[method]["label"]
        if label is None:
            continue
        scores[label] += (
            result_json[method]["score"] * weights[method]
        )
    predicted = max(scores, key=scores.get)
    return predicted, scores[predicted]


def main(image_path):
    # Classification result
    result = process_image(image_path)
    print(result)
    final_result = weighted_result(result)

    # Draw boundiong box on image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if result["yolo"]["box"] is not None:
        x1, y1, x2, y2 = result["yolo"]["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw classification result on image
        text = str(final_result[0]) + " " +  str(round(final_result[1], 2))
        position = (x1, y1 - 20)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        try:
            x, y, w, h = result["Haars Cascade"]["box"]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        except:
            pass
    
    #plt.imshow(image)
    #plt.show()
    return final_result, image

class ImageWindow:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()
        
        self.btn = tk.Button(root, text="Choose Image", command=self.load_image)
        self.btn.pack()
        
        self.name_label = tk.Label(root, text="Classification: None")
        self.name_label.pack()

        self.name_score = tk.Label(root, text="Score: None")
        self.name_score.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        image = Image.open(file_path)
        image = image.resize((600, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        self.name_label.config(text="Classification: Loading")
        self.name_score.config(text="Score: Loading")
        self.root.after(500, self.classify_image, file_path) 
        
    def classify_image(self, file_path):
        processed_image = main(file_path)
        image = Image.fromarray(processed_image[1])
        image = image.resize((600, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        self.name_label.config(text="Classification: " + processed_image[0][0])
        self.name_score.config(text="Score: " + str(round(processed_image[0][1], 2)))

if __name__ == "__main__":
    root = tk.Tk()
    window = ImageWindow(root)
    root.mainloop()
    
    #image_path = select_image()
    #main(image_path)
