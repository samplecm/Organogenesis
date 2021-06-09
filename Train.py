import matplotlib.pyplot as plt
import os
import shutil
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
import subprocess


def Train(organ,numEpochs,lr, path, processData, loadModel, preSorted):
    #processData is required if you are training with new dicom images with a certain ROI for the first time. This saves the CT and contours as image slices for training
    #loadModel is true when you already have a model that you wish to continue training
    #First extract patient training data and process it for each, saving it into Processed_Data folder

    #See if cuda is available, and set the device as either cuda or cpu if is isn't available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for training: " + device.type)
    dataPath = 'Processed_Data/' + organ + "/"
    if path==None: #if a path to data was not supplied, assume that patient data has been placed in the Patient_Files folder in the current directory. 
            patientsPath = 'Patient_Files/'
            filesFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), patientsPath)
            dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath) #this gives the absolute folder reference of the datapath variable defined above
    else: 
        filesFolder = path
        modelPath = os.path.join(path, "Models")
        dataFolder = os.path.join(path, dataPath)   
        if processData == True:
            #Now if a path has been specified, then the Model folder must be in this path as well. 
            #check for the model path:
            if not os.path.isdir(modelPath):
                #create the model path if it was not there, and refuse to load a model. 
                os.mkdir(modelPath)
                if loadModel:
                    loadModel = False 
                    modelErrorMessage = "Model directory was not found in the provided path. Model will not be loaded for training. A new model will be created in the directory.\n \
                        press enter to continue"
                    while True: #wait for user input    
                        try:
                            input = input(modelErrorMessage)
                            if input == "":
                                break
                        except KeyboardInterrupt:
                            quit()    
                        except: pass   
            #Furthermore, if processData is false, then there must exist the Processed_Data folder
            if not processData:
                dataPath = os.path.join(path, "Processed_Data")
                if not os.path.isdir(dataPath):
                    processData=True
                    preSorted=False 
                    dataErrorMessage = "Processed_Data directory was not found in the provided path. Data will have to be processed.\n \
                            press enter to continue"
                    while True: #wait for user input    
                        try:
                            input = input(dataErrorMessage)
                            if input == "":
                                break
                        except KeyboardInterrupt:
                            quit()    
                        except: pass  

            #Now run the FolderSetup.sh Script in the given directory to make sure all directories are present
            shutil.copy('FolderSetup.sh', path)
            os.chdir(path)
            subprocess.call(['sh', './FolderSetup.sh'])             
    
    DicomParsing.GetTrainingData(filesFolder, organ, preSorted, path) #go through all the dicom files and create the images
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
    iteration = 0 
    for i, (image, mask) in enumerate(val_loader):

        #validation does not require gradient calculations, turned off to reduce memory use 
        with torch.no_grad():
            image = image.to(device)
            mask = mask.to(device)
         
            loss = model.trainingStep(image,mask)
        lossHistory.append(loss.item())
        if iteration % 100 == 99:
            print("Validating on the " + str(iteration + 1) + "th image.")
            iteration += 1  

    return sum(lossHistory) / len(lossHistory)  

 
          
          
