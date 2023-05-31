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
print("Device: {}, Torch Version: {}".format(DEVICE, torch.__version__))

BATCH_SIZE = 32
EPOCHS = 10

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ]),
    'val' : transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
}

image_datasets = {x: datasets.ImageFolder("./data/hymenoptera_data",
                                          data_transforms[x]) for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size = BATCH_SIZE,
                                              num_workers = 0,
                                              shuffle = True) for x in ['train','val']}

for (X_train, y_train) in dataloaders['train']:
    print("X train: {}, type: {}".format(X_train.size(),X_train.type()))
    print("y train: {}, type: {}".format(y_train.size(),y_train.type()))
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.axis("off")
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title("Class: {}".format(str(y_train[i].item())))
plt.show()

def train(model, train_loader, optimizer, criterion, log_interval):
    model.train()
    for batch_idx,(image,label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                Epoch, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx * len(image) * len(train_loader.dataset),
                loss.item()
            ))

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output,label).item()
            prediction = output.max(1,keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,2)
summary(model,(3,224,224))
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for Epoch in range(1,EPOCHS+1):
    train(model,dataloaders["train"],optimizer,criterion,5)
    test_loss, test_accuracy = evaluate(model,dataloaders["val"])
    print("\n[Epoch: {}] \tTest Loss: {:.6f}, \tTrain Accuracy: {:.2f}%".format(
        Epoch, test_loss, test_accuracy
    ))