import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchsummary import summary

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("Device: {}, Torch Version: {}".format(DEVICE,torch.__version__))

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.CIFAR10(root="./data/CIFAR10",
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
                                 ]))

test_dataset = datasets.CIFAR10(root="./data/CIFAR10",
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
                                 ]))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size = BATCH_SIZE)

for (X_train, y_train) in train_loader:
    print("X train: {}, type: {}".format(X_train.size(),X_train.type()))
    print("y train: {}, type: {}".format(y_train.size(),y_train.type()))
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis("off")
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title("Class: {}".format(y_train[i]))
plt.show()

model = models.resnet34(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,10)
summary(model,(3,32,32))
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optimizer, criterion, log_interval):
    model.train()
    
    for batch_idx,(image,label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Epoch: {} [{}/{} ({:.0f} %)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(test_loader.dataset), 100. * batch_idx * len(image) / len(train_loader.dataset),
                loss.item()
            ))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    for image, label in test_loader:
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        output = model(image)
        test_loss += criterion(output,label).item()
        prediction = output.max(1,keepdim = True)[1]
        correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer,criterion,200)
    test_loss, test_accuracy = evaluate(model, test_loader)

    print("\n[EPOCH: {}] \tTest Loss: {}, \tTest Accuracy: {} %".format(
        Epoch, test_loss, test_accuracy
    ))