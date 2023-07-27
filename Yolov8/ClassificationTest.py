import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

model = YOLO("Yolov8/train_100_epochs_90DegreeAugSet_label_set/weights/best.pt")

model_name = "augmented90degree100Epoch"  # Name of your model
output_dir = os.path.join(os.getcwd(), f'{model_name}_results')
os.makedirs(output_dir, exist_ok=True)  # This will create the directory if it does not exist

dir_path = "Yolov8/PenguinsVTurtles/archive/valid/valid"

for filename in os.listdir(dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dir_path, filename)
        results = model(image_path)
        predictions = []  # ADDED HERE
        for result in results:
            img = result.orig_img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if result.boxes.data is not None and len(result.boxes.conf) > 0:  # Added check here
                # Find the box with the maximum confidence score
                max_i = np.argmax(result.boxes.conf)
                box = result.boxes.data[max_i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = result.names[int(result.boxes.cls[max_i])]
                score = result.boxes.conf[max_i]
                cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                predictions.append(f'{label} {score:.2f}')  # ADDED HERE

            h, w, _ = img.shape
            blank = 255 * np.ones(shape=[h//4, w, 3], dtype=np.uint8)  # ADDED HERE, adjust h//4 to increase or decrease space for text
            img = cv2.vconcat([img, blank])  # ADDED HERE
            cv2.putText(img, ', '.join(predictions), (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)  # ADDED HERE
            output_path = os.path.join(output_dir, f'testImage_{filename}')  # CHANGED HERE
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)  # ADDED HERE
            cv2.imwrite(output_path, img)  # ADDED HERE
