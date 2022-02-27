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
from PostProcessing import ThresholdRescaler
import re
import statistics
import Predict

def CrossValidate(organ, epochs=15, lr=1e-3, path=None, model_type='MultiResUNet', fold=5, dataAugmentation=True, continue_previous=False):
    """
        Performs a k-fold cross validation of the selected model type and organ.
         folds are created by linear iteration through N / k groups of patients, 
         with one group comprising the validation set in each. 
        Args:
            organ (str): the organ to train the model on 
            epochs (int): number of epochs to train each model with in each fold
            lr (float): learning rate to use during optimization
            modelType (str): type of neural network architecture to use for training
            dataAugmentation (bool): True if augmentation transforms are to be applied during training

    """
    #1. Divide the data into the k groups of train/validation data

    train_groups = []
    val_groups = []
    if path == None:
        path = os.getcwd()

    patients_folder = os.path.join(path, "Patient_Files")
    f_scores = []
    hauses = []
    current_fold=0
    cross_val_idx=None 
    first_iter=True
    while current_fold < fold:
        current_fold +=1
        if continue_previous==True and first_iter==True: 
            #if carrying off on previously started cv, need to see which fold currently on
            cv_path = models_path = os.path.join(path, "Models", organ, "CV_Models")
            num_files = len(os.listdir(cv_path))
            current_fold = int(num_files / epochs) + 1
            first_iter=False #ensure this only happens once
            load = True
            print(f"continuing cross validation on fold {current_fold} and epoch {num_files % epochs}")
        else:
            load=False    
        print(f"Beginning fold # {current_fold}")
        val_list = DicomParsing.GetTrainingData(patients_folder, organ, path, cross_val=True, fold=[fold, current_fold]) #go through all the dicom files and create the images
        #Train(organ,epochs,lr, path, modelType=model_type, processData=False, cross_val_idx=current_fold, loadModel=load)
        #temp
        if model_type.lower() == "unet":
            model = Model.UNet()
        elif model_type.lower() == "multiresunet": 
            model = Model.MultiResUNet()
        epoch_val=epochs-1    
        model_path = os.path.join(path, "Models", organ, "CV_Models", str("Model_" + model_type.lower() + "_" + organ.replace(" ", "") + "_" + "CV" + str(current_fold) + "_" + str(epoch_val) + ".pt"))    
        model.load_state_dict(torch.load(model_path))
        torch.save(model.state_dict(), os.path.join(path, "Models", organ, "Model_" + model_type.lower() + "_" + organ.replace(" ", "") + ".pt"))
        ##

        intercept = ThresholdRescaler(organ, modelType = "MultiResUNet", path=None)
        #save intercept for cross val
        saveFileName = str(current_fold) + model_type.lower() + "_" + organ.replace(" ", "") + "_rescale_intercept.txt" 
        with open(os.path.join(path, "Models/", organ, "CV_Models", saveFileName ), 'wb') as pick:
            pickle.dump(intercept, pick) 
        thres = Test.BestThreshold(organ, path, model_type, test_size=700)
        with(open(os.path.join(path, "Models", organ, "CV_Models", str(current_fold) + "Model_" + model_type.lower() + "_" + organ.replace(" ", "") + "_Thres.txt"),'wb')) as fp:
            pickle.dump(thres, fp)

        Predict.GetMultipleContours([organ], val_list ,path = None,  thresholdList = [thres]*len(val_list), modelTypeList = ["multiresunet"]*len(val_list), withReal=False, tryLoad=False, save=True)
        #f_score, recall, precision, accuracy, haus = Test.GetEvalData(organ,path,thres, modelType=model_type)

        # f_score_path = os.path.join(path, "Models", organ, "Statistics","F_Scores", "F_Score_CV_" + str(current_fold) + ".txt")
        # haus_path = os.path.join(path, "Models", organ, "Statistics","Hausdorff_Distances", "Haus_CV_" + str(current_fold) + ".txt")      
        # with open(f_score_path, "wb") as fp:
        #     pickle.dump(f_score, fp)
        # with open(haus_path, "wb") as  fp:
        #     pickle.dump(haus, fp) 

        # print(f'CV{current_fold}: F-Score: {f_score} ; H: {haus}')    
    
    #load all the f scores and hausdorff distances and take average
    # f_path = os.path.join(path, "Models", organ, "Statistics","F_Scores")
    # h_path = os.path.join(path, "Models", organ, "Statistics","Hausdorff_Distances")
    # f_files = os.listdir(f_path)
    # h_files = os.listdir(h_path)

    # for f in f_files:
    #     with open(os.path.join(f_path, f), "rb") as fp:
    #         score = pickle.load(fp)
    #         f_scores.append(score)
    # for h in h_files:
    #     with open(os.path.join(h_path, h), "rb") as fp:
    #         score = pickle.load(fp)
    #         hauses.append(score)     
               
    # avg_f = statistics.mean(f_scores)
    # avg_h = statistics.mean(hauses)

    # f_score_path = os.path.join(path, "Models", organ, "Statistics", "Final_F_Score_CV_" + str(current_fold) + ".txt")
    # haus_path = os.path.join(path, "Models", organ, "Statistics", "Final_Haus_CV_" + str(current_fold) + ".txt")
    # with open(f_score_path, "wb") as fp:
    #         pickle.dump([avg_f, f_scores], fp)
    # with open(haus_path, "wb") as  fp:
    #     pickle.dump([avg_h, hauses], fp) 

    # print(f"CV Results: \n F-Score: {avg_f} \n 95 percentile Hausdorff distance: {avg_h}")    



def Train(organ,numEpochs,lr, path, modelType, processData=True, loadModel=False, sortData=False, preSorted=False, dataAugmentation=True, cross_val_idx=None):
    """Trains a model for predicting contours on a given organ. Saves the model 
       and loss history after each epoch. Stops training after a given number of
       epochs or when the validation loss has decreased by less than 0.001 for 
       4 consecutive epochs. 

    Args:
        organ (str): the organ to train the model on 
        numEpochs (int, str): the number of epochs to train for
        lr (float, str): the learning rate to train with
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        processData (bool): True to process the data into training/testing/
            validation folders, False if they are already processed
        loadModel (bool): True to load a model and continue training, 
            False to train a new model
        modelType (str): the type of model to be used in training 
        sortData (bool): True to visually inspect contours for quality 
            assurance, False to process data without looking at contours
        preSorted(bool): True to use presorted good/bad/no contour lists, 
            False to display contours for each patient and sort manually
        dataAugmentation (bool): True to turn on data augmentation for 
            training, False to use non-augmented CT images.
        cross_val_idx (optional int): in cases where models are trained in a cross validation loop, saved weights need to include the index of the current CV iteration    
    """
    #First extract patient training data and process it for each, saving it into Processed_Data folder
    torch.cuda.empty_cache()
    dataPath = 'Processed_Data/' + organ + "/"
    if path==None: #if a path to data was not supplied, assume that patient data has been placed in the Patient_Files folder in the current directory. 
        path = pathlib.Path(__file__).parent.absolute() 
        patientsPath = 'Patient_Files/'
        filesFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), patientsPath)
        dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath) #this gives the absolute folder reference of the datapath variable defined above
        if processData:
            DicomParsing.GetTrainingData(filesFolder, organ,path, sortData, preSorted) #go through all the dicom files and create the images
            print("Data Processed")
        
    else: 
        folderScriptPath = os.path.join(pathlib.Path(__file__).parent.absolute(), "FolderSetup.sh")
        filesFolder = path
        modelPath = os.path.join(pathlib.Path(__file__).parent.absolute(), "Models")
        dataFolder = os.path.join(path, dataPath)   
        #Now if a path has been specified, then the Model folder must be in this path as well. 
            #check for the model path:           
        #Furthermore, if processData is false, then there must exist the Processed_Data folder
        if not processData:
            dataPath = os.path.join(path, "Processed_Data")
            if not os.path.isdir(dataPath): #If the processedData folder doesnt exist, then itll have to be made and data processed.
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
        else:
            shutil.copy(folderScriptPath, path)
            os.chdir(path)            
            DicomParsing.GetTrainingData(filesFolder, organ, path, sortData, preSorted) #go through all the dicom files and create the images
            subprocess.call(['sh', './FolderSetup.sh'])     
            print("Data Processed")

    #See if cuda is available, and set the device as either cuda or cpu if is isn't available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice being used for training: " + device.type)
    #Now define or load the model and optimizer: 
    epochLossHistory = []
    trainLossHistory = []
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    if loadModel == True:
        #load the model state dict for the last epoch:
        if type(cross_val_idx) == int:
            models_path = os.path.join(path, "Models", organ, "CV_Models")
            cv_model_files = sorted(os.listdir(models_path))
            if len(cv_model_files) > 0:
                nums = [int(i) for i in cv_model_files[-1] if i.isdigit()]
                current_epoch = int(str(nums[-2]) + str(nums[-1])) + 1
                if current_epoch != numEpochs:
                    model.load_state_dict(torch.load(os.path.join(models_path, cv_model_files[-1])))  
                    #get current epoch:               
                    print(f"loaded model weights for {cv_model_files[-1]}. Continuing training on Epoch {current_epoch}")
                else: 
                    current_epoch = 0    
            else:
                current_epoch = 0    
            
        else:    
            models_path = os.path.join(path, "Models", organ, "Epoch_Models")
            epoch_model_files = sorted(os.listdir(models_path))
            model.load_state_dict(torch.load(os.path.join(models_path, epoch_model_files[-1])))  
            nums = [int(i) for i in epoch_model_files[-1].split() if i.isdigit()]
            current_epoch = int(str(nums[-2]) + str(nums[-1]))+1
            print(f"loaded model weights for {epoch_model_files[-1]}. Continuing training on Epoch {current_epoch}")
        try: #try to load lists which are keeping track of the loss over time
 
            if type(cross_val_idx) == int:
                epochLossHistory = pickle.load(open(os.path.join(path, "Models", organ, "CV" + str(cross_val_idx) + "_" + modelType.lower() + "_" + "epochLossHistory" + ".txt"), "rb"))
            else:
                epochLossHistory = pickle.load(open(os.path.join(path, "Models", organ, modelType.lower() + "_" + "epochLossHistory.txt"), 'rb'))  
        except:
            
            epochLossHistory = []
    else:
        current_epoch=0    

    trainLossHistory = []       
    model.to(device)  #put the model onto the GPU     
    optimizer = torch.optim.Adam(model.parameters(), lr)

    
    dataFiles = sorted(os.listdir(dataFolder))

    #set transform = transform for data augmentation, None for no augmentation
    if dataAugmentation == True:
        transform = A.Compose ([
        A.OneOf([A.Perspective(scale=(0.05,0.1), keep_size = True, pad_mode = 0, fit_output = True, p=0.5), A.ElasticTransform(p=0.5, alpha=16, sigma=512*0.05, alpha_affine=512*0.03),
        #A.GaussNoise(var_limit = 0.05, p = 0.5)
        ], p =0.5),
        #A.OneOf([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.Rotate(5, p=0.5)], p=0.5)
        ])
    else:
        transform = None

    print("Beginning first training epoch of a " + modelType + " " + organ + " model")    
    iteration = 0
    #Criterion = F.binary_cross_entropy_with_logits()#nn.BCEWithLogitsLoss() I now just define this in the model
    epoch_range = range(current_epoch, numEpochs)
    for epoch in epoch_range:
        model.train() #put model in training mode

        #creates the training dataset 
        #set transform = transform for data augmentation, None for no augmentation
        train_dataset = CTDataset(dataFiles = dataFiles, root_dir = dataFolder, transform = transform)

        #creates the training dataloader 
        train_loader = DataLoader(dataset = train_dataset, batch_size = 1, shuffle = True)

        
        #go through all of the images in the data set 
        for i, (image, mask) in enumerate(train_loader): 
 
                image.requires_grad=True #make sure they have a gradient for training
                mask.requires_grad=True

                image = image.to(device)
                mask = mask.to(device)

                loss = model.trainingStep(image,mask) #compute the loss of training prediction

                trainLossHistory.append(loss.item())
                loss.backward() #backpropagate
                optimizer.step()
                optimizer.zero_grad()

                # if iteration % 10 == 9:
                #     print("Epoch # " + str(epoch + 1) + " --- " + "training on image #: " + str(iteration+1))
                if iteration % 100 == 99:
                    print("Epoch # " + str(epoch + 1) + " --- " + "training on image #: " + str(iteration+1) + " --- last 100 loss: " + str(sum(trainLossHistory[-100:])/100))
                iteration += 1    
       
        #end of epoch: check validation loss and
        #Save the model:
        epoch_val = str(epoch) if epoch > 9 else "0" + str(epoch)
        if type(cross_val_idx) == int:
            model_path = os.path.join(path, "Models", organ, "CV_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + "CV" + str(cross_val_idx) + "_" + epoch_val + ".pt"))
        else:    
            model_path = os.path.join(path, "Models", organ, "Epoch_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + epoch_val + ".pt"))
        torch.save(model.state_dict(), model_path) 
        
        #for param_tensor in UNetModel.state_dict():
        #    print(param_tensor, "\t", UNetModel.state_dict()[param_tensor].size())
            #if param_tensor == "multiresblock9.conv2d_bn_5x5.conv1.bias":
            #    print(UNetModel.state_dict()[0])

        #print(UNetModel.state_dict()["multiresblock1.conv2d_bn_1x1.conv1.weight"])

        #dictionary = UNetModel.state_dict()

        #for key in dictionary:
        #    print(key)

        #make a list of the hyperparameters and their labels 
        hyperparameters = []

        hyperparameters.append(["Model", modelType])
        hyperparameters.append(["Learning Rate", lr])
        hyperparameters.append(["Epochs Completed", epoch])
        hyperparameters.append(["Optimizer", "Adam"])
        hyperparameters.append(["Batch Size", "1"])
        hyperparameters.append(["Loss Function", "BCEWithLogitsLoss"])
        if dataAugmentation == True:
            hyperparameters.append(["Data Augmentation", "On"])
        else:
            hyperparameters.append(["Data Augmentation", "Off"])

        #save the hyperparameters to a binary file to be used in Test.FScore()
        with open(os.path.join(path, "Models/" + organ + "/HyperParameters_Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".txt"), "wb") as fp:
            pickle.dump(hyperparameters, fp)

        epochLoss = Validate(organ, model, path) #validation step
        epochLossHistory.append(epochLoss)
        print('Epoch # {},  Loss: {}'.format(epoch+1, epochLoss))            
                #reshape to have batch dimension in front
       
       #save the losses
        with open(os.path.join(path,"Models", organ, modelType.lower() + "_" + "epochs" + ".txt"), "wb") as fp:
            pickle.dump(epoch, fp) 
        if type(cross_val_idx) == int:
            with open(os.path.join(path, "Models", organ, "CV" + str(cross_val_idx) + "_" + modelType.lower() + "_" + "epochLossHistory" + ".txt"), "wb") as fp:
                pickle.dump(epochLossHistory, fp)     
        else:
            with open(os.path.join(path, "Models", organ, modelType.lower() + "_" + "epochLossHistory" + ".txt"), "wb") as fp:
                pickle.dump(epochLossHistory, fp)  
            
    #Now get model with lowest epoch loss and save to main model directory
    # min_loss_idx = epochLossHistory.index(min(epochLossHistory))
    # epoch_val = str(min_loss_idx) if min_loss_idx > 9 else "0" + str(min_loss_idx)
    # print(f"Lowest validation loss found during the {min_loss_idx+1}th epoch. Saving this model to the main organ model directory.")
    # if type(cross_val_idx) == int:
    #     model_path = os.path.join(path, "Models", organ, "CV_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + "CV" + str(cross_val_idx) + "_" + epoch_val + ".pt"))
    # else:    
    #     model_path = os.path.join(path, "Models", organ, "Epoch_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + epoch_val + ".pt"))
    # model.load_state_dict(torch.load(model_path))
    # torch.save(model.state_dict(), os.path.join(path, "Models", organ, "Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))
    

    epoch_val=numEpochs-1
    if type(cross_val_idx) == int:
        model_path = os.path.join(path, "Models", organ, "CV_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + "CV" + str(cross_val_idx) + "_" + epoch_val + ".pt"))
    else:    
        model_path = os.path.join(path, "Models", organ, "Epoch_Models", str("Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_" + epoch_val + ".pt"))
    model.load_state_dict(torch.load(model_path))
    torch.save(model.state_dict(), os.path.join(path, "Models", organ, "Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))

def Validate(organ, model, path=None):
    """Computes the average loss of the model on the validation data set. 

    Args:
        organ (str): the organ to train the model on 
        model (Module): the model to find the validation loss on

    Returns:
        float: the average loss on the entire validation data set

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
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
         
            loss = model.validationStep(image,mask)
        lossHistory.append(loss.item())
        if iteration % 100 == 99:
            print("Validating on the " + str(iteration + 1) + "th image.")
            iteration += 1  

    return sum(lossHistory) / len(lossHistory)  

 
          
          
