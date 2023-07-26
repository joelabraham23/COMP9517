import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

def detect_objects(image_path, model):
    results = model(image_path)
    for result in results:
        if result.boxes.data is not None and len(result.boxes.conf) > 0:
            max_i = np.argmax(result.boxes.conf)
            box = result.boxes.data[max_i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = result.names[int(result.boxes.cls[max_i])]
            score = result.boxes.conf[max_i].item()
            return label, score, (x1, y1, x2, y2)
    return None, None, None

def main(image_path):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yolov8CurrentModel.pt")
    model = YOLO(model_path)
    label, score, box = detect_objects(image_path, model)
    return label, score, box

#print(main("../CombinedMethods/valid/valid/image_id_000.jpg"))
