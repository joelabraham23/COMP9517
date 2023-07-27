## REFERENCES
# https://github.com/yazanmaalla/stereo-vision
# https://www.ijcseonline.org/pdf_paper_view.php?paper_id=3831&45-IJCSE-06281.pdf

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
import warnings

from skimage import io, util, img_as_ubyte
from enum import Enum
from typing import List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

TRAINPATH = "../FeatureClass/TurtleVPenguins/archive/train/train/"
TESTPATH = "../FeatureClass/TurtleVPenguins/archive/valid/valid/"

CLUSTERS = 1100


class Animal(Enum):
    TURTLE = 0
    PENGUIN = 1


def plot(gImg, img, kp):
    retval = img.copy()
    plt.imshow(cv.cvtColor(cv.drawKeypoints(gImg, kp, retval), cv.COLOR_BGR2RGB))
    plt.show()


def resize(imgs, h, w):
    newimgs = []
    for image in imgs:
        newimgs.append(cv.resize(image, (int(h * 0.5), int(w * 0.5))))
    return newimgs


def testSingleImage(path):
    label = -1
    # if "P" in path.split("/")[5].split(".")[0]:
    image = cv.imread(path, cv.COLOR_BGR2GRAY)
    #     label = Animal.PENGUIN.value
    # else:
    #     # image = Image(cv.imread(path + file), Animal.TURTLE.value, -1, -1)
    #     image = cv.imread(path, cv.COLOR_BGR2GRAY)
    #     label = Animal.TURTLE.value

    return image, label


def genDataset(path):
    imgs = []
    labels = []
    minH = float("inf")
    minW = float("inf")
    Dir = os.listdir(path)
    for file in Dir:
        if "P" in file.split(".")[0]:
            image = cv.cvtColor(cv.imread(path + file), cv.COLOR_BGR2GRAY)
            labels.append(Animal.PENGUIN.value)
        else:
            # image = Image(cv.imread(path + file), Animal.TURTLE.value, -1, -1)
            image = cv.imread(path + file, cv.COLOR_BGR2GRAY)
            labels.append(Animal.TURTLE.value)
        imgs.append(image)
        h = image.shape[0]
        w = image.shape[1]
        if h < minH:
            minH = h
        if w < minW:
            minW = w

    return imgs, labels, minH, minW


def genFeatures(dataset, labels, fExtractor):
    descriptors = []
    i = 0
    for image in dataset:
        if fExtractor == "SIFT":
            ext = cv.SIFT_create(300)  # this was 300 before
            # print("starting SIFT")
        elif fExtractor == "ORB":
            ext = cv.ORB_create(300)  # thie was
            # print("starting OB")

        kp = ext.detect(image, None)
        kpList = list(kp)
        kpList.sort(key=lambda x: x.response, reverse=True)
        kp, desc = ext.compute(image, tuple(kpList))
        # Make sure its not that one image thats annoying
        if desc is None or kp == 0:
            labels.pop(i)
            i += 1
            continue
        # plot(image, image, kp)

        i += 1
        if len(desc) < 125:
            while len(desc) < 125:
                desc = np.concatenate((desc, np.expand_dims(desc[0], axis=0)), axis=0)
        descriptors.append(desc)
    return np.vstack(descriptors), labels


def genSingleFeatures(image, label, fExtractor):
    descriptors = []

    if fExtractor == "SIFT":
        ext = cv.SIFT_create(300)
        # print("starting SIFT")
    elif fExtractor == "ORB":
        ext = cv.ORB_create(300)
        # print("starting OB")

    kp = ext.detect(image, None)
    kpList = list(kp)
    # kpList.sort(key=lambda x: x.response, reverse=True)
    kp, desc = ext.compute(image, tuple(kpList))
    # plot(image, image, kp)
    if len(desc) < CLUSTERS:
        while len(desc) < CLUSTERS:
            desc = np.concatenate((desc, np.expand_dims(desc[0], axis=0)), axis=0)
    descriptors.append(desc)
    return np.vstack(descriptors), label


def genKMeans(descriptors):
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42)
    retval = kmeans.fit_predict(descriptors)
    # kmeans.fit(np.array([image.desc.reshape(-1) for image in dataset]))
    return retval


def genSingleKMeans(descriptors):
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=42)
    retval = kmeans.fit_predict(descriptors)
    # kmeans.fit(np.array([image.desc.reshape(-1) for image in dataset]))
    return retval


def genHistograms(descriptors, kRetval, size):
    histograms = np.zeros((size, CLUSTERS), dtype=int)
    idx = 0
    for i in range(size):
        descListSize = len(descriptors[i])
        for j in range(descListSize):
            if idx < len(kRetval):
                index = kRetval[idx]
                histograms[i][index] += 1
                idx += 1
            else:
                break
    return histograms


def genSingleHistograms(descriptors, kRetval, size):
    histograms = np.zeros((size, CLUSTERS), dtype=int)
    idx = 0
    for i in range(size):
        descListSize = len(descriptors[i])
        for j in range(descListSize):
            if idx < len(kRetval):
                index = kRetval[idx]
                histograms[i][index] += 1
                idx += 1
            else:
                break
    return histograms


def trainClassifier(dataset, labels):
    classifiers = []
    # KNN
    print("starting KNN")
    classifier = KNeighborsClassifier(len(np.unique(labels)))
    ## SIFT
    descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 255) for th in hist]
    # knnSIFTModel =
    classifiers.append(classifier.fit(hist, labels))

    # ORB
    # print("starting ORB")
    descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 255) for th in hist]
    # knnORBModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # =========================== DT
    classifier = DecisionTreeClassifier()
    print("starting DT")
    ## SIFT
    descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 255) for th in hist]
    # dtSIFTModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # ORB
    descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 255) for th in hist]
    # dtORBModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # ============================ SGD
    # classifier = SGDClassifier()
    # print("starting SGD")
    # ## SIFT
    # descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    # kRet = genKMeans(descriptors)
    # hist = genHistograms(descriptors, kRet, len(trainlabels))
    # hist = [(th / 255) for th in hist]
    # # sgdSIFTModel = classifier.fit(hist, labels)
    # classifiers.append(classifier.fit(hist, labels))

    # # ORB
    # descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    # kRet = genKMeans(descriptors)
    # hist = genHistograms(descriptors, kRet, len(trainlabels))
    # hist = [(th / 255) for th in hist]
    # # sgdORBModel = classifier.fit(hist, labels)
    # classifiers.append(classifier.fit(hist, labels))
    return classifiers  # [knnSIFTModel knnORBModel dtSIFTModel dtORBModel ]


def testClassifier(classifiers, testData, mainLabels):
    print("start testing")
    results = []
    for i in range(0, len(classifiers), 2):
        classifier = classifiers[i]
        descriptors, labels = genSingleFeatures(testData, mainLabels, "SIFT")
        kRet = genSingleKMeans(descriptors)
        hist = genSingleHistograms(descriptors, kRet, 1)
        hist = [th / 150 for th in hist]
        # SIFTresults = classifier.predict(hist)
        results.append(classifier.predict(hist))

        classifier = classifiers[i + 1]
        descriptors, labels = genSingleFeatures(testData, mainLabels, "ORB")
        kRet = genKMeans(descriptors)
        hist = genHistograms(descriptors, kRet, 1)
        hist = [th / 150 for th in hist]
        # ORBresults = classifier.predict(hist)
        results.append(classifier.predict(hist))
    return results  # [knnSIFTResults knnORBResults dtSIFTResults dtORBResults]


def bayesianProbability(confusionMatrix):
    probabilities = np.zeros(confusionMatrix.shape[0])
    for i in range(confusionMatrix.shape[0]):
        pPositive = confusionMatrix[i, i] / np.sum(confusionMatrix[i])
        pNegative = 1 - pPositive
        pPriorPositive = np.sum(confusionMatrix[i]) / np.sum(confusionMatrix)
        pPriorNegative = 1 - pPriorPositive

        # Applying Bayesian formula: P(TURTLE|Data) = (P(Data|TURTLE) * P(TURTLE)) / P(Data)
        probabilities[i] = (pPositive * pPriorPositive) / (
            (pPositive * pPriorPositive) + (pNegative * pPriorNegative)
        )
    return probabilities


def getResults(TrainPath, TestPath):
    # generate datset and resize
    trainDataSet, trainlabels, trH, trW = genDataset(TrainPath)
    trainDataSet = resize(trainDataSet, trH, trW)

    # testDataSet, testlabels, teH, teW = genDataset(TestPath)
    # testDataSet = resize(testDataSet, teH, teW)
    testDataSet, testlabels = testSingleImage(TestPath)

    ################################### training
    classifiers = trainClassifier(trainDataSet, trainlabels)
    print(classifiers)
    ################################### testing
    results = testClassifier(classifiers, testDataSet, testlabels)

    return results  # [knnSIFTResults knnORBResults dtSIFTResults dtORBResults]


def convertResults(results):
    # cm = confusion_matrix(testlabels, results[0])
    # probabilities = bayesianProbability(cm)
    # print(f"Probability of SIFT TURTLE: {probabilities[0]:.2f}")
    # print(f"Probability of SIFT PENGUIN: {probabilities[1]:.2f}")
    # gained from testing
    # KNN
    probSiftKnnP = 0.5695
    probSiftKnnT = 0.42
    probOrbKnnP = 0.75
    probOrbKnnT = 0.50

    # DT
    probSiftDtP = 0.75
    probSiftDtT = 0.33
    probOrbDtP = 0.72
    probOrbDtT = 0.42

    penguinScore = 1.0
    turtleScore = 1.0
    if results[0][0] == 0:
        # TURTLE
        turtleScore = turtleScore * probSiftKnnT
    else:
        # PENGUIN
        penguinScore = penguinScore * probSiftKnnP

    if results[1][0] == 0:
        # TURTLE
        turtleScore = turtleScore * probOrbKnnT
    else:
        # PENGUIN
        penguinScore = penguinScore * probOrbKnnP

    if results[2][0] == 0:
        # TURTLE
        turtleScore = turtleScore * probSiftDtT
    else:
        # PENGUIN
        penguinScore = penguinScore * probSiftDtP

    if results[3][0] == 0:
        # TURTLE
        turtleScore = turtleScore * probOrbDtT
    else:
        # PENGUIN
        penguinScore = penguinScore * probOrbDtP

    if penguinScore > turtleScore:
        return "penguin", penguinScore
    else:
        return "turtle", turtleScore
    # result = [knnSIFTResults knnORBResults dtSIFTResults dtORBResults]


def main(testFilePath):
    #################################################
    # generate datset and resize
    # trainDataSet, trainlabels, trH, trW = genDataset(TRAINPATH)
    # trainDataSet = resize(trainDataSet, trH, trW)

    # # testDataSet, testlabels, teH, teW = genDataset(TESTPATH)
    # # testDataSet = resize(testDataSet, teH, teW)
    # testDataSet, testlabels = testSingleImage(testFilePath)
    results = getResults(TRAINPATH, testFilePath)
    # print(convertResults(results))
    return convertResults(results)


# main("FeatureClass/TurtleVPenguins/archive/valid/valid/P1.jpg")
# print("==============KNN===================")
# print(f"Accuracy score for SIFT: {accuracy_score(testlabels, results[0])}")
# cm = confusion_matrix(testlabels, results[0])
# probabilities = bayesianProbability(cm)
# print(f"Probability of SIFT TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of SIFT PENGUIN: {probabilities[1]:.2f}")


# print(f"Accuracy score for ORB: {accuracy_score(testlabels, results[1])}")
# cm = confusion_matrix(testlabels, results[1])
# probabilities = bayesianProbability(cm)
# print(f"Probability of orb TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of orb PENGUIN: {probabilities[1]:.2f}")

# print("==============DT===================")
# print(f"Accuracy score for SIFT: {accuracy_score(testlabels, results[2])}")
# cm = confusion_matrix(testlabels, results[2])
# probabilities = bayesianProbability(cm)
# print(f"Probability of SIFT TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of SIFT PENGUIN: {probabilities[1]:.2f}")


# print(f"Accuracy score for ORB: {accuracy_score(testlabels, results[3])}")
# cm = confusion_matrix(testlabels, results[3])
# probabilities = bayesianProbability(cm)
# print(f"Probability of orb TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of orb PENGUIN: {probabilities[1]:.2f}")

# print("==============SGD===================")
# print(f"Accuracy score for SIFT: {accuracy_score(testlabels, results[4])}")
# cm = confusion_matrix(testlabels, results[4])
# probabilities = bayesianProbability(cm)
# print(f"Probability of SIFT TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of SIFT PENGUIN: {probabilities[1]:.2f}")


# print(f"Accuracy score for ORB: {accuracy_score(testlabels, results[5])}")
# cm = confusion_matrix(testlabels, results[5])
# probabilities = bayesianProbability(cm)
# print(f"Probability of orb TURTLE: {probabilities[0]:.2f}")
# print(f"Probability of orb PENGUIN: {probabilities[1]:.2f}")
