import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time

from skimage import io, util, img_as_ubyte
from enum import Enum
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

TRAINPATH = "FeatureClass/TurtleVPenguins/archive/train/train/"
TESTPATH = ""
start = time.time()


class Animal(Enum):
    TURTLE = 0
    PENGUIN = 1


class Image:
    def __init__(self, img, animal, kp, desc):
        self.img = img
        self.gImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.animal = animal
        self.kp = kp
        self.desc = desc


def SIFT(img):
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(img, None)
    return kp, desc


def plotSIFT(gImg, img, kp):
    retval = img.copy()
    return cv.drawKeypoints(gImg, kp, retval)


def resize(dataSet, h, w):
    for image in dataSet:
        image.img = cv.resize(image.img, (h, w))
        image.gImg = cv.resize(image.gImg, (h, w))
    return dataSet


def genSIFT(dataSet):
    for image in dataSet:
        image.kp, image.desc = SIFT(image.gImg)
    return dataSet


# trains model, returns it
def trainKNN(trainingData):
    knn = KNeighborsClassifier(
        n_neighbors=25,
        weights="distance",
        algorithm="auto",
        leaf_size=50,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    )
    knn.fit(
        np.array([image.img.reshape(-1) for image in trainingData]),
        np.array([image.animal for image in trainingData]),
    )
    return knn


def populateDataset(path):
    dataset = []
    minH = float("inf")
    minW = float("inf")
    trainDir = os.listdir(path)
    for file in trainDir:
        image = None
        if "P" in file.split("."[0]):
            image = Image(cv.imread(path + file), Animal.PENGUIN.value, 0, 0)

        else:
            image = Image(cv.imread(path + file), Animal.TURTLE.value, 0, 0)

        h, w, _ = image.img.shape
        if h < minH:
            minH = h
        if w < minW:
            minW = w

        dataset.append(image)
    return dataset, minH, minW


def testModel(model, testData):
    return model.predict(np.array([image.img.reshape(-1) for image in testData]))


dataset, h, w = populateDataset(TRAINPATH)
dataset = genSIFT(resize(dataset, int(h * 0.1), int(w * 0.1)))
knn = trainKNN(dataset)
prediction = testModel(knn, dataset)
trainLabels = np.array([image.animal for image in dataset])
print(accuracy_score(trainLabels, prediction))
