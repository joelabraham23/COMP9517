import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import glob


class DenseNet(torch.nn.Module):
    def __init__(self, inputs, num_hid, outputs):
        super(DenseNet, self).__init__()
        self.layer1 = nn.Linear(inputs, num_hid)
        self.layer1to2 = nn.Linear(inputs, num_hid)
        self.layer1to3 = nn.Linear(inputs, outputs)
        self.layer2 = nn.Linear(num_hid, num_hid)
        self.layer2to3 = nn.Linear(num_hid, outputs)
        self.layer3 = nn.Linear(num_hid, outputs)

        # first 2 use tan
        self.tanh = nn.Tanh()
        # last use sigmoid
        self.sig = nn.Sigmoid()

    def forward(self, input):
        output = self.layer1(input)
        inputTo2 = self.layer1to2(input)
        inputTo3 = self.layer1to3(input)
        output = self.tanh(output)
        self.hid1 = output
        hid1To3 = self.layer2to3(self.hid1)

        output = self.layer2(output)
        output = self.tanh(output + inputTo2)
        self.hid2 = output
        # output
        output = self.layer3(output)
        output = self.sig(output + inputTo3 + hid1To3)
        return output


def train(net, train_loader, optimizer):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # cv2.drawKeypoints(img, keypoint[:10], img, color=(255, 0, 0))
        # plt.imshow(img)
        # plt.show()
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


net = DenseNet(len(labels[0]["descriptor"]), 50, 1)

if list(net.parameters()):
    # initialize weight values
    for we in list(net.parameters()):
        # starting weights
        we.data.normal_(0, 0.08)

    # use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005, weight_decay=0.0001)

    # training loop
    epoch = 0
    count = 0
    max_epoch = 20000
    while epoch < max_epoch and count < 2000:
        epoch = epoch + 1
        accuracy = train(net, train_loader, optimizer)
        if accuracy == 100:
            count = count + 1
        else:
            count = 0
torch.save(net, "./cannySiftNN/briefnn.pt")
