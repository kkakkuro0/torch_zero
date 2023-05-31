import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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

print(f"device: {DEVICE}, torch version: {torch.__version__}")

BATCH_SIZE = 32
EPOCHS = 10

train_dataset = datasets.CIFAR10(root = "./data/CIFAR10",
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "./data/CIFAR10",
                                 train=False,

                                 transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False)

for (X_train, y_train) in train_loader:
    print(f"X train: {X_train.size()}, type: {X_train.type()}") 
    print(f"y train: {y_train.size()}, type: {y_train.type()}") 
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis("off")
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title(f"Class: {str(y_train[i].item())}")
# plt.show()

class MLP(nn.Module):
    def __init__(self) :
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(32*32*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,10)

    def forward(self,x):
        x = x.view(-1,32*32*3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)
        return x
    
model = MLP().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
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
            print("Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx*len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
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
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}% \n".format(
        Epoch, test_loss, test_accuracy
    ))
