import matplotlib.pyplot as plt
import os
import pathlib
import glob
import pickle
import numpy as np
import random
import math
import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, dataset 
from shapely.geometry import Point
import torch.nn.functional as F
import DicomParsing
import Model
import Test
from Dataset import CTDataset
import albumentations as A 


def Train(organ,numEpochs,lr, processData=False, loadModel=False):
    #processData is required if you are training with new dicom images with a certain ROI for the first time. This saves the CT and contours as image slices for training
    #loadModel is true when you already have a model that you wish to continue training
    #First extract patient training data and process it for each, saving it into Processed_Data folder

    #See if cuda is available, and set the device as either cuda or cpu if is isn't available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for training: " + device.type)

    dataPath = 'Processed_Data/' + organ + "/"
    if processData == True:
        patientsPath = 'Patient_Files/'
        DicomParsing.GetTrainingData(patientsPath, organ, preSorted = False) #go through all the dicom files and create the images
        print("Data Processed")
    #Now define or load the model and optimizer: 
    epochLossHistory = []
    trainLossHistory = []
    UNetModel = Model.UNet()
    if loadModel == True:
        UNetModel.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))  
        try: #try to load lists which are keeping track of the loss over time
            trainLossHistory = pickle.load(open(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Loss History/" + organ + "/")), str(trainLossHistory)), 'rb'))  
            epochLossHistory = pickle.load(open(os.path.join(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Loss History/" + organ + "/")), str(epochLossHistory)), 'rb'))  
        except:
            trainLossHistory = []
            epochLossHistory = []
    UNetModel.to(device)  #put the model onto the GPU     
    optimizer = torch.optim.Adam(UNetModel.parameters(), lr)

    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath) #this gives the absolute folder reference of the datapath variable defined above
    dataFiles = sorted(os.listdir(dataFolder))

    transform = A.Compose ([
    A.OneOf([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.Rotate(20, p=0.5)], p=0.5),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    ])

    print("Beginning Training")    
    iteration = 0
    #Criterion = F.binary_cross_entropy_with_logits()#nn.BCEWithLogitsLoss() I now just define this in the model
    
    for epoch in range(numEpochs):
        UNetModel.train() #put model in training mode

        #creates the training dataset 
        #set transform = transform for data augmentation, None for no augmentation
        train_dataset = CTDataset(dataFiles = dataFiles, root_dir = dataFolder, transform = None)

        #creates the training dataloader 
        train_loader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True)

        #go through all of the images in the data set 
        for i, (image, mask) in enumerate(train_loader): 
                
                image.requires_grad=True #make sure they have a gradient for training
                mask.requires_grad=True

                image = image.to(device)
                mask = mask.to(device)
                loss = UNetModel.trainingStep(image,mask) #compute the loss of training prediction
                trainLossHistory.append(loss.item())
                loss.backward() #backpropagate
                optimizer.step()
                optimizer.zero_grad()

                if iteration % 10 == 9:
                    print("Epoch # " + str(epoch + 1) + " --- " + "training on image #: " + str(iteration+1))
                if iteration % 100 == 99:
                    print("Epoch # " + str(epoch + 1) + " --- " + "training on image #: " + str(iteration+1) + " --- last 100 loss: " + str(sum(trainLossHistory[-100:])/100))
                iteration += 1    
       
        #end of epoch: check validation loss and
        #Save the model:
        torch.save(UNetModel.state_dict(), os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")) 
        
        #make a list of the hyperparameters and their labels 
        hyperparameters = []

        hyperparameters.append(["Model", "UNet"])
        hyperparameters.append(["Learning Rate", lr])
        hyperparameters.append(["Epochs Completed", epoch])
        hyperparameters.append(["Optimizer", "Adam"])
        hyperparameters.append(["Batch Size", "1"])
        hyperparameters.append(["Loss Function", "BCEWithLogitsLoss"])
        hyperparameters.append(["Data Augmentation", "Off"])

        #save the hyperparameters to a binary file to be used in Test.FScore()
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/HyperParameters_Model_" + organ.replace(" ", "") + ".txt"), "wb") as fp:
            pickle.dump(hyperparameters, fp)

        epochLoss = Validate(organ, UNetModel) #validation step
        epochLossHistory.append(epochLoss)
        print('Epoch # {},  Loss: {}'.format(epoch+1, epochLoss))            
                #reshape to have batch dimension in front
       
       #save the losses
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Loss History/" + organ + "/" + "trainLossHistory" + ".txt")), "wb") as fp:
            pickle.dump(trainLossHistory, fp)         
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Loss History/" + organ + "/" + "epochLossHistory" + ".txt")), "wb") as fp:
            pickle.dump(sum(epochLossHistory)/len(epochLossHistory), fp)  

         

def Validate(organ, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath)
    dataFiles = sorted(os.listdir(dataFolder))
    lossHistory = []
    print('Validating')

    #creates the validation dataset 
    val_dataset = CTDataset(dataFiles = dataFiles, root_dir = dataFolder, transform = None)

    #creates the validation dataloader 
    val_loader = DataLoader(dataset = val_dataset, batch_size = 1, shuffle = True)

    for i, (image, mask) in enumerate(val_loader):

        #validation does not require gradient calculations, turned off to reduce memory use 
        with torch.no_grad():
            image = image.to(device)
            mask = mask.to(device)
         
            loss = model.trainingStep(image,mask)
        lossHistory.append(loss.item())

    return sum(lossHistory) / len(lossHistory)  

 
          
          
