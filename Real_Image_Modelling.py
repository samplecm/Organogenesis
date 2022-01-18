import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from torchvision import datasets
import torchvision.transforms as transforms


class Model(nn.Module):
    def __init(self):
        super(Model, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(28*28, 20), #number pixels for input
            nn.ReLU(),
            nn.Conv2d(20, 64),
            nn.ReLU(),
            nn.Conv2d(64,128),
            nn.ReLU(),
            nn.Conv2d(128,64),
            nn.ReLU(),
            nn.Conv2d(64,20),
            nn.ReLU(),
            nn.Conv2d(20,10)#10 outputs

        )
    def forward(self, image):
        out = self.sequence(image)
        return out
def Get_MNIST():
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_data, [55000, 5000])
    train_loader = DataLoader(train, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)
    return train_loader, val_loader

    print("")

def Train():
    print("Training Model")    
    model = Model()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  
    loss = nn.CrossEntropyLoss()

    train_loader, val_loader = Get_MNIST()

    
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch in train_loader:
            x, y = batch
            #x = batch size x 1 x pixel x pixel
            x = x.view(x.size(0), -1)

            logits = model(x)
            J = loss(logits, y)

            model.zero_grad()
            J.backward()

            optimizer.step()


if __name__ == "__main__":
    Train()


        
        


