import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import glob
import os

from densenet import DenseNet


def train(net, train_loader, optimizer, device, epoch):
    total = 0
    correct = 0
    for batch_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # zero the gradients
        data.to(device)
        output = net(data)  # apply network
        loss = F.binary_cross_entropy(output, target)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights
        pred = (output >= 0.5).float()
        correct += (pred == target).float().sum()
        total += target.size()[0]
        accuracy = 100 * correct / total

    if epoch % 100 == 0:
        print("ep:%5d loss: %6.4f acc: %5.2f" % (epoch, loss.item(), accuracy))

    return accuracy


def extractFeatures(img, detector, computer, debug):
    keypoint = detector.detect(img, None)
    if isinstance(keypoint, tuple):
        keypoint = list(keypoint)
    keypoint.sort(key=lambda x: x.response, reverse=True)
    keypoint, descriptor = computer.compute(img, keypoint)
    descriptor = descriptor[:12]
    if len(descriptor) < 12 and not debug:
        while len(descriptor) < 12:
            descriptor = np.concatenate((descriptor, [descriptor[0]]), axis=0)
    return keypoint, descriptor



def setupNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "./cannySiftNN/train/*.jpg"
    train_annot_path = "./cannySiftNN/train_annotations"
    with open(train_annot_path, "r") as f:
        labels = json.loads(f.read())

    i = 0
    star = cv2.xfeatures2d.StarDetector_create(responseThreshold=15)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    debugFeatureExtract = False
    for file in glob.glob(path):
        img = cv2.imread(file)

        keypoint, labels[i]["descriptor"] = extractFeatures(
            img, star, brief, debugFeatureExtract
        )

        # debug and show images with lack of descriptors
        if len(labels[i]["descriptor"]) < 12:
            print("img_" + str(i) + " keypoints_" + str(len(labels[i]["descriptor"])))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.drawKeypoints(img, keypoint[:10], img, color=(255, 0, 0))
            plt.imshow(img)
            plt.show()
            while len(labels[i]["descriptor"]) < 12:
                labels[i]["descriptor"] = np.concatenate(
                    (labels[i]["descriptor"], [labels[i]["descriptor"][0]]), axis=0
                )
        labels[i]["descriptor"] = labels[i]["descriptor"].flatten()
        i += 1
    print(len(labels[i - 1]["descriptor"]))

    ## NN

    train_data = [data["descriptor"] / 255 for data in labels]
    target = [[data["category_id"] - 1] for data in labels]
    train_data = torch.tensor(train_data, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_data, target)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__()
    )

    net = DenseNet(len(labels[0]["descriptor"]), 8, 1)

    if list(net.parameters()):
        # initialize weight values
        for we in list(net.parameters()):
            # starting weights
            we.data.normal_(0, 0.008)

        # use Adam optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.0007)

        # training loop
        epoch = 0
        count = 0
        max_epoch = 200000
        accuracy = 0
        while epoch < max_epoch and (epoch < 10000 or count < 2000):
            epoch = epoch + 1
            accuracy = train(net, train_loader, optimizer, device, epoch)
            if accuracy > 99:
                count += 1
            else:
                count = 0

    torch.save(net, "./cannySiftNN/briefnn.pt")


def validateNN():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load("./cannySiftNN/briefnn.pt")
    path = "./cannySiftNN/valid/*.jpg"
    valid_annot_path = "./cannySiftNN/train_annotations"
    with open(valid_annot_path, "r") as f:
        labels = json.loads(f.read())

    star = cv2.xfeatures2d.StarDetector_create(responseThreshold=15)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    debugFeatureExtract = False
    i = 0
    penguinTrue = 0
    penguinFalse = 0
    turtleTrue = 0
    turtleFalse = 0
    accuracy = 0
    for file in glob.glob(path):
        img = cv2.imread(file)
        keypoint, descriptor = extractFeatures(img, star, brief, debugFeatureExtract)
        descriptor = descriptor.flatten()
        valid_data = descriptor / 255
        valid_data = torch.tensor(valid_data, dtype=torch.float32)
        resultClass = round(net.forward(valid_data).item())
        trueClass = labels[i]["category_id"] - 1

        # penguin
        if trueClass == 0:
            if trueClass == resultClass:
                penguinTrue += 1
                accuracy += 1
            else:
                penguinFalse += 1
        else:
            if trueClass == resultClass:
                turtleTrue += 1
                accuracy += 1
            else:
                turtleFalse += 1

        i += 1
    confusion = [[penguinTrue / i, penguinFalse / i], [turtleTrue / i, turtleFalse / i]]
    accuracy = accuracy / i
    print("[[pT, pF],[tT, tF]]")
    print(confusion)
    print("accuracy = " + str(accuracy) + "%")


def main(filePath):
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "briefnn.pt")
    net = torch.load(model_path)
    penguinCertainty = 0.692
    turtleCertaninty = 0.666
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = cv2.imread(filePath)

    star = cv2.xfeatures2d.StarDetector_create(responseThreshold=15)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    debugFeatureExtract = False

    keypoint, descriptor = extractFeatures(img, star, brief, debugFeatureExtract)
    descriptor = descriptor.flatten()
    valid_data = descriptor / 255
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    resultClass = round(net.forward(valid_data).item())

    if resultClass == 0:
        animal = "penguin"
        certainty = penguinCertainty
    else:
        animal = "turtle"
        certainty = turtleCertaninty
    return animal, certainty

#print(main("../CombinedMethods/valid/valid/image_id_003.jpg"))

