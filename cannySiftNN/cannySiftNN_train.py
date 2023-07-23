import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import json
import glob


path = "./cannySiftNN/train/*.jpg"
train_annot_path = "./cannySiftNN/train_annotations"
with open(train_annot_path, "r") as f:
    labels = json.loads(f.read())

i = 0
star = cv2.xfeatures2d.StarDetector_create(responseThreshold=15)
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
things = []
for file in glob.glob(path):
    img = cv2.imread(file)
    keypoint = star.detect(img, None)
    keypoint.sort(key=lambda x: x.response, reverse=True)
    keypoint, descriptor = brief.compute(img, keypoint)
    labels[i]["descriptor"] = descriptor[:12]
    things.append(len(keypoint))
    if len(labels[i]["descriptor"]) < 12:
        print("img_" + str(i) + " keypoints_" + str(len(labels[i]["descriptor"])))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.drawKeypoints(img, keypoint[:10], img, color=(255, 0, 0))
        plt.imshow(img)
        plt.show()
        while len(labels[i]["descriptor"]) < 12:
            labels[i]["descriptor"].append(labels[i]["descriptor"][0])
    i += 1
