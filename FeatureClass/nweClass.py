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


def stackList(list):
    stack = np.array(list[0])
    for items in list[1:]:
        stack = np.vstack((stack, items))
    return stack


def SIFT(image):
    sift = cv.SIFT_create()

    kp = []
    # OPTIMISATION: change step, lower = higher density of kp, higher = lower density of kp
    (
        h,
        w,
    ) = image.gImg.shape
    for i in range(STEP, w, STEP):
        for j in range(STEP, h, STEP):
            kp.append(cv.KeyPoint(i, j, STEP))
    kp, des = sift.compute(image.gImg, kp)
    # plot(image.gImg, image.img, kp)
    return kp, des


def ORB(image):
    orb = cv.ORB_create(nfeatures=1500)

    kp = []
    # OPTIMISATION: change step, lower = higher density of kp, higher = lower density of kp
    (
        h,
        w,
    ) = image.gImg.shape
    for i in range(STEP, w, STEP):
        for j in range(STEP, h, STEP):
            kp.append(cv.KeyPoint(i, j, STEP))
    kp, des = orb.compute(image.gImg, kp)
    return kp, des


# unavailbale due to copyright
# def SURF(image):
#     surf = cv.SURF_create()

#     kp = []
#     # OPTIMISATION: change step, lower = higher density of kp, higher = lower density of kp
#     (
#         h,
#         w,
#     ) = image.gImg.shape
#     for i in range(STEP, w, STEP):
#         for j in range(STEP, h, STEP):
#             kp.append(cv.KeyPoint(i, j, STEP))
#     kp, des = surf.compute(image.gImg, kp)
#     return kp, des


def genKMeans(dataset):
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42)
    stackedList = stackList(np.array([image.desc for image in dataset]))
    retval = kmeans.fit_predict(stackedList)
    # kmeans.fit(np.array([image.desc.reshape(-1) for image in dataset]))
    return kmeans, retval


def genHistograms(dataset, kRetval):
    histograms = np.array([np.zeros(CLUSTERS) for i in range(len(dataset))])
    count = 0
    for i in range(len(dataset)):
        descListSize = len(dataset[i].desc)
        for j in range(descListSize):
            index = kRetval[descListSize + j]
            histograms[i][index] += 1
        count += descListSize
    return histograms


def plot(gImg, img, kp):
    retval = img.copy()
    plt.imshow(cv.cvtColor(cv.drawKeypoints(gImg, kp, retval), cv.COLOR_BGR2RGB))
    plt.show()


# generate sift features of an image
def genFeatures(dataset, classifier):
    for image in dataset:
        if classifier == "SIFT":
            image.kp, image.desc = SIFT(image)
        # elif classifier == "SURF":
        #     image.kp, image.desc = SURF(image)
        elif classifier == "ORB":
            image.kp, image.desc = ORB(image)
    return dataset


def resize(dataSet, h, w):
    for image in dataSet:
        image.img = cv.resize(image.img, (int(h * 0.7), int(w * 0.7)))
        image.gImg = cv.resize(image.gImg, (int(h * 0.7), int(w * 0.7)))
    return dataSet


# trains KNN model, returns it
def trainKNN(dataset):
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
    # KNN on SIFT
    knnTrainingSet = genFeatures(dataset, "SIFT")
    kmeans, kRetval = genKMeans(knnTrainingSet)
    knnTrainingHistograms = genHistograms(knnTrainingSet, kRetval)
    knnSIFT = knn.fit(
        knnTrainingHistograms, np.array([image.animal for image in knnTrainingSet])
    )

    # KNN on ORB
    knnTrainingSet = genFeatures(dataset, "ORB")
    kmeans, kRetval = genKMeans(knnTrainingSet)
    knnTrainingHistograms = genHistograms(knnTrainingSet, kRetval)
    knnORB = knn.fit(
        knnTrainingHistograms, np.array([image.animal for image in knnTrainingSet])
    )

    return knnSIFT, knnORB


# populate a dataset with Image objects, SIFT features are generated in here
def populateDataset(path):
    dataset = []
    minH = float("inf")
    minW = float("inf")
    trainDir = os.listdir(path)
    for file in trainDir:
        image = None

        if "P" in file.split(".")[0]:
            image = Image(cv.imread(path + file), Animal.PENGUIN.value, -1, -1)

        else:
            image = Image(cv.imread(path + file), Animal.TURTLE.value, -1, -1)

        h, w, _ = image.img.shape
        if h < minH:
            minH = h
        if w < minW:
            minW = w

        dataset.append(image)
    return resize(dataset, minH, minW)


def testKNN(knn, testData):
    sift, surf = knn
    knnSIFTTestSet = genFeatures(testData, "SIFT")
    knnORBTestSet = genFeatures(testData, "ORB")

    # SIFT
    kmeans, kRetval = genKMeans(knnSIFTTestSet)
    # predicting based on histograms
    SIFTtestingHistograms = genHistograms(knnSIFTTestSet, kRetval)

    # SURF
    kmeans, kRetval = genKMeans(knnORBTestSet)
    # predicting based on histograms
    ORBtestingHistograms = genHistograms(knnORBTestSet, kRetval)
    return sift.predict(SIFTtestingHistograms), surf.predict(ORBtestingHistograms)


dataset = populateDataset(TRAINPATH)
testDataSet = populateDataset(TESTPATH)
dataset = shuffle(dataset, random_state=42)
testDataSet = shuffle(testDataSet, random_state=42)

# # KNN on SIFT
# knnTrainingSet = genFeatures(dataset, "SIFT")
# # knnTestSet = genFeatures(testDataSet, "SIFT")

# kmeans, kRetval = genKMeans(knnTrainingSet)
# knnTrainingHistograms = genHistograms(knnTrainingSet, kRetval)

SIFTpredictionKNN, ORBpredictionKNN = testKNN(trainKNN(dataset), testDataSet)

testLabels = np.array([image.animal for image in testDataSet])
print(f"Accuracy score for SIFT: {accuracy_score(testLabels, SIFTpredictionKNN)}")
print(f"Accuracy score for ORB: {accuracy_score(testLabels, ORBpredictionKNN)}")
# print(f"confusionn matric for SIFT: \n{confusion_matrix(testLabels, predictionKNN)}")
