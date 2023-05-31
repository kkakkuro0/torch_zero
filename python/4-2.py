import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("device: {}, torch version: {}".format(DEVICE,torch.__version__))

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.CIFAR10(root="./data/CIFAR10",
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)

test_dataset = datasets.CIFAR10(root="./data/CIFAR10",
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=False)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)

for (X_train, y_train) in train_loader:
    print("X train: {}, type: {}".format(X_train.size(),X_train.type()))
    print("t train: {}, type: {}".format(y_train.size(),y_train.type()))
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis("off")
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title("Class: {}".format(str(y_train[i].item())))
# plt.show()

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.fc1 = nn.Linear(8*8*16,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1,8*8*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x
    
model = CNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

# print(model)

def train(model,train_loader,optimizer,criterion,log_interval):
    model.train()
    for batch_idx,(image,label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if batch_idx%log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain loss = {}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx * len(image) / len(train_loader.dataset),
                loss.item()
            ))

def evaluate(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image,label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output,label).item()
            prediction = output.max(1,keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(1,EPOCHS+1):
    train(model,train_loader,optimizer,criterion,200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[Epoch :{}], \tTest Loss: {:.4f}, \tTest accuracy: {:.2f}%\n".format(
        Epoch, test_loss, test_accuracy
    ))
