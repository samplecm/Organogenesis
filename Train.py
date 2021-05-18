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
        DicomParsing.GetTrainingData(patientsPath, organ) #go through all the dicom files and create the images
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

    print("Beginning Training")    
    iteration = 0
    #Criterion = F.binary_cross_entropy_with_logits()#nn.BCEWithLogitsLoss() I now just define this in the model
    
    for epoch in range(numEpochs):
        UNetModel.train() #put model in training mode
        #Loop through the patients, in a random order
        filesRange = list(range(len(dataFiles)))
        random.shuffle(filesRange)
        
        d = 0
        while d < len(filesRange):
            #loading 200 images at a time for training so that RAM isn't totally filled up
            numStack = min(200, len(filesRange) - 1 - d)
            p = 0
            concatList = []
            while p < numStack:
                imagePath = dataFiles[d]
                image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
                image = image.reshape((2,1,image.shape[1], image.shape[2]))
                #add to a list of all these training images
                concatList.append(image)
                p+=1
                d+=1
            if len(concatList) == 0:
                break    
            data = np.concatenate(concatList, axis=1)    #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
            print('Loaded ' + str(data.shape[1]) + ' images for training. Proceeding...')          
            data = torch.from_numpy(data)
            numSlices = data.shape[1]
            slices = list(range(numSlices))
            random.shuffle(slices)
            #now images are in a random order and we begin training
            for sliceNum in slices:
                # x = torch.from_numpy(data[0, sliceNum, :, :]).cuda()
                # y = torch.from_numpy(data[1:3, sliceNum, :,:]).cuda()
                x = data[0, sliceNum, :, :] #CT image
                y = data[1, sliceNum, :, :] #binary mask
                
                xLen, yLen = x.shape

                x.requires_grad=True #make sure they have a gradient for training
                y.requires_grad=True
                #need to reshape 
                x = torch.reshape(x, (1,1,xLen,yLen)).float()
                y = torch.reshape(y, (1,1,xLen,yLen)).float()
                x = x.to(device)
                y = y.to(device)
                loss = UNetModel.trainingStep(x,y) #compute the loss of training prediction
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
    d = 0
    print('Validating')
    while d < len(dataFiles):
        numStack = min(200, len(dataFiles) - 1 - d) #loading 1400 images at a time (takes about 20GB RAM)
        p = 0
        concatList = []
        while p < numStack:
            imagePath = dataFiles[d]
            image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
            image = image.reshape((2,1,image.shape[2], image.shape[2]))
            concatList.append(image)
            p+=1
            d+=1
        if len(concatList) == 0:
            break    
        data = np.concatenate(concatList, axis=1)  
        numSlices = data.shape[1]
        for sliceNum in range(numSlices):
            x = torch.from_numpy(data[0, sliceNum, :, :])
            y = torch.from_numpy(data[1:2, sliceNum, :,:])
            x = x.to(device)
            y = y.to(device)
            xLen, yLen = x.shape
            #need to reshape 
            x.requires_grad = True
            y.requires_grad = True
            x = torch.reshape(x, (1,1,xLen,yLen)).float()
            y = torch.reshape(y, (1,1,xLen,yLen)).float()
            loss = model.trainingStep(x,y)
            lossHistory.append(loss.item())
    return sum(lossHistory) / len(lossHistory)        
          
          
