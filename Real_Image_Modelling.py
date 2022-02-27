import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split, dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.nn import Linear, Conv2d, MaxPool2d, ReLU, LogSoftmax
import statistics
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, num_channels, classes):
        super(Model, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, image):
        x = self.conv1(image)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
def Get_MNIST():
    train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_split = 0.9
    val_split = 1- train_split
    num_train_samples = int(len(train_data)* train_split)
    num_val_samples = len(train_data) - num_train_samples


    train, val = random_split(train_data, [num_train_samples, num_val_samples], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, shuffle=True, batch_size=32)
    val_loader = DataLoader(val, batch_size=32)
    return train_loader, val_loader



def Train():
    print("Training Model")  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    model = Model(3, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3)  
    loss = nn.CrossEntropyLoss()


    train_loader, val_loader = Get_MNIST()
    train_steps = len(train_loader.dataset) // 32
    val_steps = len(val_loader.dataset) // 32
    H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}
    
    num_epochs = 30
    for epoch in range(num_epochs):
        train_correct = 0
        val_correct = 0
        train_loss = 0
        val_loss = 0
        for batch in train_loader:
            x, y = batch
            #x = batch size x 1 x pixel x pixel
            # x = x.view(x.size(0), -1)

            x = x.repeat(1,3,1,1) #convert to rgb 
            x = x.to(device)
            
            y = y.to(device)
            logits = model(x)
            
            J = loss(logits, y)


            model.zero_grad()
            J.backward()

            optimizer.step()
            train_loss += J.item()
            train_correct += (logits.argmax(1) == y).type(torch.float).sum().item()

        avg_train_loss = train_loss / train_steps
        #print(f"Training # correct: {avg_correct_train}")    
        print(f"Training loss: {avg_train_loss}")
        H["train_loss"].append(avg_train_loss)
        H["train_acc"].append(train_correct / len(train_loader.dataset))

        #validation loop
        losses = []    
        for batch in val_loader:
            x,y = batch
            b = x.size(0)    
            # x = x.view(b, -1)
            x = x.repeat(1,3,1,1) #convert to rgb 
            x = x.to(device)
            
            y = y.to(device)
            with torch.no_grad():
                logits = model(x)
            J = loss(logits, y)    
            val_loss += J.item()
            val_correct += (logits.argmax(1) == y).type(torch.float).sum().item()
        #print(f"Val # correct: {val_correct / val_steps}")    
        print(f"Val loss: {val_loss / val_steps}")
        H["val_loss"].append(val_loss/ val_steps)
        H["val_acc"].append(val_correct / len(val_loader.dataset))


    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train loss")
    plt.plot(H["val_loss"], label="validation loss")
    plt.plot(H["train_acc"], label="train set accuracy")
    plt.plot(H["val_acc"], label="validation set accuracy")
    plt.title("Loss/Accuracy History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    Train()


        
        


