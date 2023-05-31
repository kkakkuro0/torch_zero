import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("device: {}, torch version: {}".format(DEVICE,torch.__version__))

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

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = BATCH_SIZE)

for (X_train,y_train) in train_loader:
    print("X train: {}, type: {}".format(X_train.size(),X_train.type()))
    print("y train: {}, type: {}".format(y_train.size(),y_train.type()))
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis("off")
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title("Class: {}".format(y_train[i].item()))
plt.show()

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNET(nn.Module):
    def __init__(self, num_classes = 10) -> None:
        super(ResNET,self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16,2,stride=1)
        self.layer2 = self._make_layer(32,2,stride=2)
        self.layer3 = self._make_layer(64,2,stride=2)
        self.linear = nn.Linear(64,num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes,planes,stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out,8)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out
    
model = ResNET().to(DEVICE)
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
            print("Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
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

    print("\n[EPOCH: {}] \tTest Loss: {}, \tTest Accuracy: {}".format(
        Epoch, test_loss, test_accuracy
    ))