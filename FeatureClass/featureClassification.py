## REFERENCES
# https://github.com/yazanmaalla/stereo-vision
# https://www.ijcseonline.org/pdf_paper_view.php?paper_id=3831&45-IJCSE-06281.pdf

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time

from skimage import io, util, img_as_ubyte
from enum import Enum
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

TRAINPATH = "FeatureClass/TurtleVPenguins/archive/train/train/"
TESTPATH = "FeatureClass/TurtleVPenguins/archive/valid/valid/"
CLUSTERS = 648
STEP = 100


class Animal(Enum):
    TURTLE = 0
    PENGUIN = 1


def resize(imgs, h, w):
    newimgs = []
    for image in imgs:
        newimgs.append(cv.resize(image, (int(h * 0.5), int(w * 0.5))))
    return newimgs


def genDataset(path):
    imgs = []
    labels = []
    minH = float("inf")
    minW = float("inf")
    Dir = os.listdir(path)
    for file in Dir:
        if "P" in file.split(".")[0]:
            image = cv.imread(path + file)
            labels.append(Animal.PENGUIN.value)
        else:
            # image = Image(cv.imread(path + file), Animal.TURTLE.value, -1, -1)
            image = cv.imread(path + file)
            labels.append(Animal.PENGUIN.value)
        imgs.append(image)
        h, w, _ = image.shape
        if h < minH:
            minH = h
        if w < minW:
            minW = w

    return imgs, labels, minH, minW


def genFeatures(dataset, labels, classifier):
    descriptors = []
    if classifier == "SIFT":
        classi = cv.SIFT_create(40)
    elif classifier == "ORB":
        classi = cv.ORB_create(15)
    i = 0
    for image in dataset:
        kp, desc = classi.detectAndCompute(cv.cvtColor(image, cv.COLOR_BGR2GRAY), None)
        if desc is None or kp == 0:
            labels.pop(i)
            i += 1
            continue
        descriptors.append(desc)
        i += 1
    return np.array(descriptors, dtype=object), labels


#################################################
# generate datset and resize
trainDataSet, trainlabels, trH, trW = genDataset(TRAINPATH)
trainDataSet = resize(trainDataSet, trH, trW)

# generate FEATURES
descriptors, trainlabels = genFeatures(trainDataSet, trainlabels, "SIFT")
descriptors = np.vstack(descriptors)

crit = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
flags = cv.KMEANS_RANDOM_CENTERS
compactness, labels, centres = cv.kmeans(descriptors, 150, None, crit, 10, flags)

def bag_of_features(features, centres, k = 500):
      vec = np.zeros((1, k))
      for i in range(features.shape[0]):
          feat = features[i]
          diff = np.tile(feat, (k, 1)) - centres
          dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
          idx_dist = dist.argsort()
          idx = idx_dist[0]
          vec[0][idx] += 1
      return vec

labels = []
vec = []
for descriptor in descriptors:
    img_vec = bag_of_features(descriptor, centres, 150)
    vec.append(img_vec)
vec = np.vstack(vec)

knn = KNeighborsClassifier(
        n_neighbors=35,
        weights="distance",
        algorithm="auto",
        leaf_size=50,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    )

knnSIFT = knn.fit(
        vec, np.array(labels)
    )
