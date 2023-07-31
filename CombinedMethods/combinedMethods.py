import json
import sys
import cv2
import matplotlib.pyplot as plt
import os
import tkinter as tk
import numpy as np
import statistics

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

#0 is bad, 1 is best IOU calculation
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w1, h1 = w1 + x1, h1 + y1
    w2, h2 = w2 + x2, h2 + y2
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = min(w1, w2) - x_intersection
    h_intersection = min(h1, h2) - y_intersection
    area_of_intersection = w_intersection * h_intersection
    area_of_box1 = (w1 - x1) * (h1 - y1)
    area_of_box2 = (w2 - x2) * (h2 - y2)
    area_of_union = area_of_box1 + area_of_box2 - area_of_intersection
    iou_score = area_of_intersection / area_of_union
    return iou_score

def main(image_path):
    # Classification result
    result = process_image(image_path)

    final_result = weighted_result(result)

    # Draw boundiong box on image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if result["yolo"]["box"] is not None:
        x, y, w, h = result["yolo"]["box"]
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 3)
        
        # Draw classification result on image
        text = str(final_result[0]) + " " +  str(round(final_result[1], 2))
        position = (x, y - 20)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        try:
            x, y, w, h = result["Haars Cascade"]["box"]
            w = x + w
            h = y + h
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 3)
        except:
            pass
    
    #plt.imshow(image)
    #plt.show()
    
    return final_result, image, [x, y, w, h], image_path

class ImageWindow:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=600, height=600)
        self.canvas.pack()
        
        self.btn = tk.Button(root, text="Choose validation annotation file", command=self.load_val_file)
        self.btn.pack()
        
        self.btn = tk.Button(root, text="Choose Image", command=self.load_image)
        self.btn.pack()
        
        self.or_label = tk.Label(root, text="OR")
        self.or_label.pack()

        self.btn = tk.Button(root, text="Choose Folder", command=self.load_folder)
        self.btn.pack()

        self.name_label = tk.Label(root, text="Classification: None")
        self.name_label.pack()

        self.name_score = tk.Label(root, text="Score: None")
        self.name_score.pack()
        
        self.IoU_score = tk.Label(root, text="IoU: None")
        self.IoU_score.pack()

        self.distance_score = tk.Label(root, text="Pred to Val center distance: None")
        self.distance_score.pack()
        
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.iou_scores = []
        self.distance_scores = []

    def calculate_metrics(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

        if self.tp > 0:
            precision = self.tp / (self.tp + self.fp)
            recall = self.tp / (self.tp + self.fn)
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f1_score = 0, 0, 0

        print(self.tp , self.tn , self.fp , self.fn)

        tp_p = self.tp
        tn_p = self.tn
        fp_p = self.fp
        fn_p = self.fn

        print()
        print('Confusion Matrix:')
        print(f'TP (Penguin): {tp_p}, FP (Penguin): {fp_p}')
        print(f'FN (Turtle) : {fn_p}, TN (Turtle) : {tn_p}')
        print()
        print(f'\nMetrics:')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1_score}')

    def load_val_file(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, 'r') as f:
            self.data = json.load(f)
            
    def load_folder(self):
        self.file_list = []
        path = filedialog.askdirectory()
        for filename in os.listdir(path):
            if filename.lower().endswith((".jpg")):
                file_path = os.path.join(path, filename)
                self.file_list.append(file_path)
        for file_path in self.file_list:
            self.load_image(file_path)

    def load_image(self, filepath=None):
        self.canvas.delete("all")
        if filepath is None:
            self.file_path = filedialog.askopenfilename()
        else:
            self.file_path = filepath

        self.image = Image.open(self.file_path)
        self.image = self.image.resize((600, 600), Image.ANTIALIAS)
        self.image = np.array(self.image)
        
        self.name_label.config(text="Classification: Loading")
        self.name_score.config(text="Score: Loading")
        self.IoU_score.config(text="IoU: Loading")
        self.distance_score.config(text="Pred to Val center distance: Loading")

        self.root.after(500, self.classify_image, self.file_path)

        
    def classify_image(self, file_path):
        processed_image = main(file_path)
        self.image = np.array(processed_image[1])
        self.image_pil = Image.fromarray(self.image)
        self.image_pil = self.image_pil.resize((600, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        self.name_label.config(text="Classification: " + processed_image[0][0])
        self.name_score.config(text="Score: " + str(round(processed_image[0][1], 2)))
        self.root.after(1, self.validate_classification, processed_image) 

    def print_iou_distance(self):
        print()
        print("IoU Scores: ")
        #for score in self.iou_scores:
        #    print(score)
        if self.iou_scores: 
            print("IoU mean: ", statistics.mean(self.iou_scores))
            print("IoU standard deviation: ", statistics.stdev(self.iou_scores))
        print()
        print("Distance Scores: ")
        #for score in self.distance_scores:
        #    print(score)
        if self.distance_scores: 
            print("Distance mean: ", statistics.mean(self.distance_scores))
            print("Distance standard deviation: ", statistics.stdev(self.distance_scores))
            
    def validate_classification(self, results):
        
        image_id_cur = str(int(results[3].split('_')[2].split('.')[0]))
        image_val_data = (list(filter(lambda obj: obj["id"] == int(image_id_cur), self.data))[0])
        
        #predicted bounding box
        x_pred, y_pred, w_pred, h_pred = results[2]
        
        #Drawing validation bounding box in red
        x_val, y_val, w_val, h_val = image_val_data["bbox"]
        w_val = x_val+w_val
        h_val = y_val+h_val
        cv2.rectangle(self.image, (x_val, y_val), (w_val, h_val), (255, 0, 0), 1)
        
        self.image_pil = Image.fromarray(self.image)
        self.image_pil = self.image_pil.resize((600, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(self.image_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        #calculate distance of center of predicted and validation box 
        center_val = [x_val + w_val / 2, y_val + h_val / 2]
        center_pred = [x_pred + w_pred / 2, y_pred + h_pred / 2]
        distance = np.linalg.norm(np.array(center_val) - np.array(center_pred))
        iou_score = iou([x_val, y_val, w_val, h_val], [x_pred, y_pred, w_pred, h_pred])

        self.iou_scores.append(iou_score)
        self.distance_scores.append(distance)
        
        self.IoU_score.config(text="IoU: " + str(iou_score))
        self.distance_score.config(text="Pred to Val center distance: " + str(distance))

        #Calculating performance metrics
        #This is 1 or 2, 1 for penguin and 2 for turtle 
        class_label_map = {"penguin": 1, "turtle": 2}

        predicted_label = class_label_map[results[0][0]]
        correct_label = image_val_data["category_id"]
        
        print()
        print("predicted")
        print(results[0],predicted_label)
        print("real result")
        print(correct_label)
        print()
        if predicted_label == correct_label == 1:
            self.tp += 1
            print("##tp##")
        elif predicted_label == correct_label == 2:
            self.tn += 1
            print("##tn##")
        elif predicted_label == 1 and correct_label == 2:
            self.fp += 1
            print("##fp##")
        elif predicted_label == 2 and correct_label == 1:
            self.fn += 1
            print("##fn##")
        
if __name__ == "__main__":
    root = tk.Tk()
    window = ImageWindow(root)
    root.mainloop()
    window.calculate_metrics()
    window.print_iou_distance()
    
