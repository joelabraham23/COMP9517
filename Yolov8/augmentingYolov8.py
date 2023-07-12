import os
import cv2
import numpy as np
import time

base_path = "G:/Uni/Year4/Comp9517/Ass2/turtleVpenguin.v1i.yolov8_Original-90-deg-aug/"
dataset_parts = ['train', 'valid', 'test']


start_time = time.time()

print("Started")
for part in dataset_parts:
    images_dir = os.path.join(base_path, part, 'images')
    labels_dir = os.path.join(base_path, part, 'labels')

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(images_dir, filename)
            img = cv2.imread(img_path)
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(img_path.replace('.jpg', '_rot90.jpg'), img_rotated)

            label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))
            with open(label_path, 'r') as f:
                labels = f.readlines()
            with open(label_path.replace('.txt', '_rot90.txt'), 'w') as f:
                for label in labels:
                    cls, x_center, y_center, width, height = map(float, label.strip().split())
                    x_center_rot, y_center_rot = 1 - y_center, x_center
                    width_rot, height_rot = height, width
                    f.write(f"{int(cls)} {x_center_rot:.6f} {y_center_rot:.6f} {width_rot:.6f} {height_rot:.6f}\n")
            print(filename)

end_time = time.time()
print(f"Done and dusted in {end_time - start_time:.2f} seconds.")
