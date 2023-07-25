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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

TRAINPATH = "FeatureClass/TurtleVPenguins/archive/train/train/"
TESTPATH = "FeatureClass/TurtleVPenguins/archive/valid/valid/"
CLUSTERS = 1000
# STEP = 100


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


def SIFT(image):
    return


def ORB(image):
    return


def genFeatures(dataset, labels, fExtractor):
    descriptors = []
    i = 0
    for image in dataset:
        if fExtractor == "SIFT":
            ext = cv.SIFT_create(10)
            print("starting SIFT")
        elif fExtractor == "ORB":
            ext = cv.ORB_create(10)
            print("starting OB")

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
        if len(desc) < 5:
            while len(desc) < 5:
                desc = np.concatenate((desc, np.expand_dims(desc[0], axis=0)), axis=0)
        descriptors.append(desc)
    return np.vstack(descriptors), labels


def genKMeans(descriptors):
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


def trainClassifier(dataset, labels):
    classifiers = []
    # KNN
    print("starting KNN")
    classifier = KNeighborsClassifier(len(np.unique(labels)))
    ## SIFT
    descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # knnSIFTModel =
    classifiers.append(classifier.fit(hist, labels))

    # ORB
    print("starting ORB")
    descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # knnORBModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # =========================== DT
    classifier = DecisionTreeClassifier()
    print("starting DT")
    ## SIFT
    descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # dtSIFTModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # ORB
    descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # dtORBModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # ============================ SGD
    classifier = SGDClassifier()
    print("starting SGD")
    ## SIFT
    descriptors, trainlabels = genFeatures(dataset, labels, "SIFT")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # sgdSIFTModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))

    # ORB
    descriptors, trainlabels = genFeatures(dataset, labels, "ORB")
    kRet = genKMeans(descriptors)
    hist = genHistograms(descriptors, kRet, len(trainlabels))
    hist = [(th / 127) for th in hist]
    # sgdORBModel = classifier.fit(hist, labels)
    classifiers.append(classifier.fit(hist, labels))
    return classifiers  # [knnSIFTModel knnORBModel dtSIFTModel dtORBModel sgdSIFTModel sgdORBModel]


def testClassifier(classifiers, testData, mainLabels):
    results = []
    for i in range(0, len(classifiers), 2):
        classifier = classifiers[i]
        descriptors, labels = genFeatures(testData, mainLabels, "SIFT")
        kRet = genKMeans(descriptors)
        hist = genHistograms(descriptors, kRet, len(labels))
        hist = [th / 127 for th in hist]
        # SIFTresults = classifier.predict(hist)
        results.append(classifier.predict(hist))

        classifier = classifiers[i + 1]
        descriptors, labels = genFeatures(testData, mainLabels, "ORB")
        kRet = genKMeans(descriptors)
        hist = genHistograms(descriptors, kRet, len(labels))
        hist = [th / 127 for th in hist]
        # ORBresults = classifier.predict(hist)
        results.append(classifier.predict(hist))
    return results  # [knnSIFTResults knnORBResults dtSIFTResults dtORBResults sgdSIFTResults sgdORBResults]


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

    testDataSet, testlabels, teH, teW = genDataset(TestPath)
    testDataSet = resize(testDataSet, teH, teW)

    ################################### training
    classifiers = trainClassifier(trainDataSet, trainlabels)
    ################################### testing
    results = testClassifier(classifiers, testDataSet, testlabels)

    return results  # [knnSIFTResults knnORBResults dtSIFTResults dtORBResults sgdSIFTResults sgdORBResults]


#################################################
# generate datset and resize
trainDataSet, trainlabels, trH, trW = genDataset(TRAINPATH)
trainDataSet = resize(trainDataSet, trH, trW)

testDataSet, testlabels, teH, teW = genDataset(TESTPATH)
testDataSet = resize(testDataSet, teH, teW)
results = getResults(TRAINPATH, TESTPATH)

print(f"Accuracy score for SIFT: {accuracy_score(testlabels, results[0])}")
cm = confusion_matrix(testlabels, results[0])
probabilities = bayesianProbability(cm)
print(f"Probability of SIFTKNN TURTLE: {probabilities[0]:.2f}")
print(f"Probability of SIFTKNN PENGUIN: {probabilities[1]:.2f}")


print(f"Accuracy score for SIFT: {accuracy_score(testlabels, results[1])}")
cm = confusion_matrix(testlabels, results[1])
probabilities = bayesianProbability(cm)
print(f"Probability of orbKNN TURTLE: {probabilities[0]:.2f}")
print(f"Probability of orbKNN PENGUIN: {probabilities[1]:.2f}")
