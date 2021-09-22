#!/usr/bin/env python

#local dependencies:
from math import nan
import DicomParsing
import Train
import Test
import Predict
import PostProcessing
import re 
import argparse 
import pickle 
import pathlib
import os 
import DicomSaving 

#Create a dictionary of organs and regular expressions for organs
organOps ={
    "Body": re.compile(r"body?"), 
    "Spinal Cord": re.compile(r"(spi?n?a?l)?(-|_| )?cord"),
    "Oral Cavity": re.compile(r"or?a?l(-|_| )?cavi?t?y?"),
    "Left Parotid": re.compile(r"le?f?t?(-|_| )?(par)o?t?i?d?"),
    "Right Parotid": re.compile(r"ri?g?h?t?(-|_| )?(par)o?t?i?d?"),
    "Left Submandibular": re.compile(r"le?f?t(-|_| )?subma?n?d?i?b?u?l?a?r?"),
    "Right Submandibular": re.compile(r"ri?g?h?t?(-|_| )?subma?n?d?i?b?u?l?a?r?"),
    "Brainstem": re.compile(r"b?r?a?i?n?stem"),
    "Larynx": re.compile(r"lary?n?(x|g?o?p?h?a?r?y?n?x?)?"),
    "Brain": re.compile(r"brain"),
    "Brachial Plexus": re.compile(r"brachi?a?l?(-|_| )?Plexu?s?"),
    "Chiasm": re.compile(r"chia?s?m?"),
    "Esophagus": re.compile(r"esopha?g?u?s?"),
    "Globes": re.compile(r"globe?s?"),
    "Lens": re.compile(r"lens"),
    "Lips": re.compile(r"lips?"),
    "Mandible": re.compile(r"mand?ible?"),
    "Optic Nerves": re.compile(r"opti?c?(-|_| )?nerv?e?"),
    "Left Tubarial": re.compile(r"le?f?t?(-|_| )?(tub)a?r?i?a?l?"),
    "Right Tubarial": re.compile(r"ri?g?h?t?(-|_| )?(tub)a?r?i?a?l?"),
    "All": re.compile(r"all")
}
#Create a list of possible functions
functionOps = [
    "Train",
    "GetContours",
    "BestThreshold",
    "GetEvalData",
    "PlotMasks",
    "GetEvalData"
]

def main():
    print("Welcome to Organogenesis")
    print("------------------")

    #Keep a list of available structures for training/predicting
    OARs = ["Body", "Spinal Cord", "Oral Cavity", "Left Parotid", "Right Parotid", "Left Submandibular", "Right Submandibular", "Brainstem","Left Tubarial", "Right Tubarial", "All"] 

    #Need to get user input. Make a string to easily ask for a number corresponding to an OAR.
    ChooseOAR_string = "Please enter the number(s) for the organ(s) you wish to contour / train a model for. Separate the numbers with spaces \n>>"
    for i in range(len(OARs)):
        ChooseOAR_string += str(i + 1) + ": " + OARs[i] + "\n" #create list of options
    
    while True: #wait for user input    
        try:
            chosenOARsNums = input(ChooseOAR_string)
            chosenOARsNums = chosenOARsNums.split(" ")
            chosenOARs = []
            OARCheck = True
            for num in chosenOARsNums:
                if int(num) > len(OARs) or int(num) < 1:
                    OARCheck = False
            if OARCheck == True:
                for num in chosenOARsNums:
                    chosenOARs.append(OARs[int(num)-1])
                if "All" in chosenOARs:
                    chosenOARs = []
                    for OAR in OARs:
                        if OAR != "All":
                            chosenOARs.append(OAR)
                print("\nSelected Organ(s):")
                for i, OAR in enumerate(chosenOARs):
                    print(str(i+1) + "." + OAR)
                break
        except KeyboardInterrupt:
            quit()    
        except: pass     

    #Now determine if the goal is to train or to find contours, etc
    chooseTask_string = "\nPlease enter the number for the desired task\n"
    chooseTask_string += "1. Train a model for predicting "
    chooseTask_string += "\n2. Predict contours using an existing model"
    chooseTask_string += "\n3. Determine the best threshold accuracy for predicting"
    chooseTask_string += "\n4. Determine the evaulation data (F score and 95th percentile Haussdorff distance) for the validation set"
    chooseTask_string += "\n5. Plot predicted masks"
    chooseTask_string += "\n6. Export model to ONNX" #Not worrying about this anymore for now
    chooseTask_string += "\n7. predict using ONNX model \n>>" #Not worrying about this anymore for now
    
    while True: #get user input
        try:
            task = int(input(chooseTask_string))
            if (task in range(0,8)):
                break
        except KeyboardInterrupt:
            quit()
        except: pass   

    if (task == 1):
        Train.Train(chosenOARs[0], 10, 1e-4, path=None, processData=False, loadModel=False, modelType = "UNet", sortData=False, preSorted=False)
        #Test.Best_Threshold(OARs[chosenOAR],400)
        #Test.TestPlot(OARs[chosenOAR], path=None, threshold=0.1)  

    elif task == 2:    
        Predict.GetMultipleContours(chosenOARs,"2",path = None,  thresholdList = [0.1], modelTypeList = ["unet"], withReal=True, tryLoad=False) 
        
    elif task == 3:
        Test.BestThreshold(chosenOARs[0], path=None, testSize=300, modelType = "unet")
    elif task == 4:
        F_Score, recall, precision, accuracy, haussdorffDistance = Test.GetEvalData(chosenOARs[0], threshold=0., path = None, modelType = "multiresunet")    
        print([F_Score, recall, precision, accuracy, haussdorffDistance])
        
    elif task == 5:
        #array, y = Test.GetMasks(OARs[chosenOAR], "P10", path=None, threshold=0.7, modelType = "UNet")
        #import numpy as np
        #print(np.amax(y))
        ##print(np.amax(array))
        #Test.TestPlot(chosenOARs[0], path=None, threshold=0.7, modelType = "UNet") 
        Test.PercentStats(chosenOARs[0], path = None)

if __name__ == "__main__":
    
        
    parser = argparse.ArgumentParser(
        description="Organogenesis: an open source program for autosegmentation of medical images"
    )
    parser.add_argument('-o', "--organs", help="Specify organ(s) to train/evaluate a model for or predict/generate contours with. Include a single space between organs. \
        Please choose from:\n body, \n brain, \n brainstem, \n brachial-plexus, \n chiasm, \n esophagus, \n globes, \n larynx, \n lens, \n lips, \n mandible, \n optic-nerves, \n oral-cavity, \n right-parotid, \n left-parotid, \n spinal-cord,\n right-submandibular, \n left-submandibular, \n right-tubarial, \n left-tubarial,\n all\n", nargs = '+', default=None, type = str)
    parser.add_argument('-f', "--function", help = "Specify the function to be performed. Options include \"Train\": to train a model for predicting the specified organ, \
        \"GetContours\": to obtain predicted contours for a patient, \"BestThreshold\": to find the best threhold for maximizing a model's F score, \"GetEvalData\": to calculate the F score, recall, precision, accuracy and 95th percentile Haussdorff distance for the given organ's model, \
        \"PlotMasks\": to plot 2d CTs with both manually drawn and predicted masks for visual comparison", default=None, type=str)
    #Training parameters:    
    parser.add_argument("--lr", help="Specify the learning rate desired for model training", default=None, type=float)
    parser.add_argument("--epochs", help="Specify the number of epochs to train the model for", default=None, type=int)
    parser.add_argument("--processData", help="True/False. True if patient DICOM data needs to be processed into training/validation/test folders", default=False, action='store_true')
    parser.add_argument("--loadModel", help="True/False. True if a pre-existing model is to be loaded to continue training", default=False, action='store_true')
    parser.add_argument("--dataPath", help="If data is not prepared in Patient_Files folder, specify the path to the directory containing all patient directories",type=str, default=None)
    parser.add_argument("--preSorted", help="True/False. True if contours have been sorted into \"good\" and \"bad\" contour lists and the data should be processed into test/validation/training folders using them, False if not", default=False, action='store_true')
    parser.add_argument("--modelType", help="Specify the model type. UNet or MultiResUNet. If predicting with multiple organs, please enter the model types in the same order as the organs separated by a single space", type=str, default=None, nargs = '+')
    parser.add_argument("--dataAugmentation", help="True/False. True to turn on data augmentation for training, False to use non-augmented CT images", default=False, action='store_true')
    parser.add_argument("--sortData", help="True/False. True if the patient list is to be visually inspected for quality assurance of the contours, False if confident that all contours are well contoured", default=False, action='store_true')
    #GetContours parameters:
    parser.add_argument("--predictionPatientName", help= "Specify the name of the patient folder in the Patient_Files folder that you wish to predict contours for. Alternatively, supply the full path to a patient's folder",type=str, default=None)
    parser.add_argument("--thres", help="Specify the pixel mask threshold to use with the model (between 0 and 1). If predicting with multiple organs, please enter the thresholds in the same order as the organs separated by a single space", type=float, default=None, nargs = '+')
    parser.add_argument("--contoursWithReal", help="True/False. True to plot the predicted contours alongside manually contoured ones from the patient's dicom file, False to just plot the predicted contours", default=False , action='store_true')
    parser.add_argument("--loadContours", help="True/False. True to attempt to load previously predicted or processed contours to save time, False to predict or process data without trying to load files", default=False, action='store_true')
    parser.add_argument("--dontSaveContours", help="True/False. Specify whether or not you would like the predicted contours saved to a DICOM file. True to not save predicted contours, False to save them", default=False, action='store_true')

    args = parser.parse_args()
    v = vars(args)
    n_args = sum([ 1 for a in v.values( ) if a])


    if (n_args == 0):
        main()

    print("Welcome to Organogenesis")
    print("------------------")

    #Now ensure that a proper organ has been supplied. 
    organs = ""
    organs = args.organs

    if organs != None:
        organsList = []
        organMatch = False
        for i, organ in enumerate(organs): 
            organ = organ.lower()
            organs[i] = organ
            for key in organOps: 
                if (re.match(organOps[key], organ)): 
                    organsList.append(key) 
                    break

        if len(organsList) == len(organs):
            organMatch = True

    if (organs == None) or (organMatch==False):
        while True: #get user input
            organMatch = False 
            try:
                organsSelected = input("\nInvalid or no organ(s) specified. Please specify the organ(s) that you wish to train/evaluate a model for or predict/generate contours with separated by a single space.\n\nPlease choose from: \n body, \n brain, \n brainstem, \n brachial-plexus, \n chiasm, \n esophagus, \n globes, \n larynx, \n lens, \n lips, \n mandible, \n optic-nerves, \n oral-cavity, \n right-parotid, \n left-parotid, \n spinal-cord,\n right-submandibular, \n left-submandibular, \n all\n>")
                organs = list(organsSelected.split(" "))
                organsList = []
                for i, organ in enumerate(organs): 
                    organ = organ.lower()
                    organs[i] = organ
                    for key in organOps: 
                        if (re.match(organOps[key], organ)): 
                            organsList.append(key) 
                            break
                if len(organsList) == len(organs):
                    organMatch = True
                    break
            except KeyboardInterrupt:
                quit()
            except: pass  

    #Add all organs if all was selcted and print selected organs
    if "All" in organsList:
        organsList = []
        for key in organOps:
            if key != "All":
                organsList.append(key)
    organsList = list(dict.fromkeys(organsList)) #remove duplicates
    print("\nSelected organ(s): ")
    for i, organ in enumerate(organsList): 
        print(str(i+1) + "." + str(organ))

    #Now ensure that a proper function has been requested         
    functionMatch = False 
    for function in functionOps:
        if function == args.function:
            functionMatch = True 
            break
    if not functionMatch:
        while True: #get user input 
            try:
                functionSelection = input("\nInvalid function or no function specified. Please specify the function to be performed. Options include: \n\"Train\": to train a model for predicting the specified organ, \n\"GetContours\": Obtain predicted contour point clouds for a patient, \n\"BestThreshold\": find the best threhold for maximizing the model's F score, \n\"GetEvalData\": calculate the F score, recall, precision, accuracy and 95th percentile Haussdorff distance for the given organ's model, \n\"PlotMasks\": plot 2d CTs with both manually drawn and predicted masks for visual comparison \n >")
                for function in functionOps:
                    if (function == functionSelection):
                        args.function = functionSelection
                        print("\nSelected function: " + functionSelection)
                        functionMatch = True
                        organ = key
                        break
                if functionMatch:
                    break    
            except KeyboardInterrupt:
                quit()
            except: pass  
    else: 
        print("\nSelected function: " + args.function)        

    #Now perform the specified function:
    if args.function == "Train":
        if len(organsList) > 1:
            print("\nTraining can only be done with one organ at a time. Training will proceed for the " + organsList[0] + "\n")
        #Get learning rate
        lr = args.lr
        if (lr == None):
            while True:
                try:
                    lr = input("\nPlease specify the learning rate\n >")
                    lr = float(lr)
                    break    
                except KeyboardInterrupt:
                    quit()
                except: pass  
        #Get number of epochs
        numEpochs = args.epochs
        if (numEpochs == None):
            while True:
                try:
                    numEpochs = input("\nPlease specify the number of epochs\n >")
                    numEpochs = int(numEpochs)
                    break    
                except KeyboardInterrupt:
                    quit()
                except: pass  

        modelType = args.modelType

        #if multiple model types were provided, chooses the first one
        if modelType is not None:
            if isinstance(modelType, list):
                modelType = modelType[0]
                if modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                    modelType = None
            elif modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                modelType = None

        if (modelType == None):
            while True:
                try:
                    modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                    modelType = str(modelType)
                    if " " in modelType:
                        modelType = modelType.split(" ")[0]
                        if modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                            break
                    elif modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                        break  
                except KeyboardInterrupt:
                    quit()
                except: pass  

        processData = args.processData
        loadModel = args.loadModel
        dataPath = args.dataPath #If dataPath is None, then the program uses the data in the patient_files folder. If it is a path to a directory, data will be processed in this directory. 
        sortData = args.sortData
        preSorted = args.preSorted
        dataAugmentation = args.dataAugmentation

        Train.Train(organsList[0], numEpochs, lr, dataPath, processData, loadModel, modelType, sortData, preSorted, dataAugmentation)
        bestThreshold = Test.BestThreshold(organsList[0], dataPath, modelType = modelType, testSize = 400)

        Test.TestPlot(organsList[0], dataPath, threshold=bestThreshold, modelType = modelType)  

    elif args.function == "GetContours":

        path = args.dataPath    
        if path == None:
            thresLoadPath = pathlib.Path(__file__).parent.absolute() 
        else: 
            thresLoadPath = path    
        patient = args.predictionPatientName
        if (patient == None):
            while True:
                try:
                    patient = input("\nPlease specify the name of the patient folder that you are trying to get contours for\n >")
                    if patient != "":
                        break    
                except KeyboardInterrupt:
                    quit()
                except: pass
                
        modelType = args.modelType
        modelTypeList = []

        modelType = args.modelType

        if modelType is not None:
            if isinstance(modelType, list):
                if len(modelType) != len(organsList) and len(modelType) != 1: #make sure the correct number of model types were given 
                    modelType = None
                else: 
                    for model in modelType: 
                        if model.lower() != "unet" and model.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                            modelType = None
                            break
            elif modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                modelType = None

        if (modelType == None):
            while True:
                try:
                    modelType = input("\nPlease specify the model type (UNet or MultiResUNet). If predicting with multiple organs, please enter the model types in the same order as the organs separated by a space. If a single model type will be used for all organs, just enter it once. \n >")      
                    if " " in modelType:
                        modelCheck = True
                        modelTypeList = list(modelType.split(" "))
                        for model in modelTypeList: 
                            if model.lower() != "unet" and model.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                                modelCheck = False
                                break
                        if modelCheck == True:
                            if len(modelTypeList) == len(organsList) or len(modelTypeList) == 1: #make sure the correct number of model types were given
                                break
                    else:
                        if modelType.lower() == "unet" or modelType.lower() == "multiresunet": #make sure that the model is either unet or multiresunet
                            modelTypeList = [modelType]
                            break
                except KeyboardInterrupt:
                    quit
                except: pass 
        elif isinstance(modelType, list): 
            modelTypeList = modelType
        else:
            modelTypeList = [modelType]

        #if only one model type is being used for predicting on multiple organs 
        if len(organsList) > 1 and len(modelTypeList) == 1:
            modelType = modelTypeList[0]
            modelTypeList = []
            for organ in organsList:
                modelTypeList.append(modelType)

        thres = args.thres     
        if thres == None:
            organs_wo_bestThres = []
            thresList = []
            for i in range(len(organsList)):#try to load best thresholds for organ and model
                #initialize size of thres to match organsList
                thresList.append(None) 
                try:  
                    bestThresFile = open(str(os.path.join(thresLoadPath, "Models/Model_" + modelTypeList[i].lower() + "_" + organsList[i].replace(" ", "")) + "_Thres.txt"), "r")   
                    bestThres = bestThresFile.read()
                    bestThresFile.close()
                    thresList[-1] = float(bestThres)
                    print("\nBest threshold of " + str(thresList[i]) + " loaded for " + modelTypeList[i] + " " + organsList[i] + " predictions")
                except: 
                    pass
            #Now check which ones had best threshold    

            for i in range(len(organsList)):
                if thresList[i] == None:
                    #check to see if there is any validation data
                    #if there isn't, then best threshold cannot be run and user must input a threshold
                    dataPath = 'Processed_Data/' + organsList[i] + "_Val/"
                    dataFolder = os.path.join(thresLoadPath, dataPath)
                    dataFiles = os.listdir(dataFolder)
                    if len(dataFiles) > 0:
                        print("\nBest threshold has not been determined for " + modelTypeList[i] + " " + organsList[i] + " predictions. Launching the BestThreshold function.")
                        thresList[i] = Test.BestThreshold(organsList[i], path, modelTypeList[i], 500)
                    else: 
                        while True:
                                try:
                                    thresList[i] = float(input("\nBest threshold could not be found for the " + organsList[i] + " " + modelTypeList[i] + " model. Please specify a threshold to use for predicting \n >"))
                                    break    
                                except KeyboardInterrupt:
                                    quit()
                                except: pass        
                        
        elif (len(thres) != len(organsList)):
            while True:
                try:
                    thres = input("\nInvalid thresholds were supplied. Please specify the threshold to be used for contour prediction(s). If predicting with multiple organs, please enter the thresholds in the same order as the organs separated by a space\n >")      
                    thresList = list(thres.split(" "))
                    if len(thresList) == len(organsList):
                        break
                except KeyboardInterrupt:
                    quit
                except: pass     
        else: 
            thresList = thres
        for i, threshold in enumerate(thresList):
            thresList[i] = float(threshold)
        tryLoad = args.loadContours
        withReal = args.contoursWithReal   
        dontSave = args.dontSaveContours

        if dontSave == False:
            save = True
        else: 
            save = False

        combinedContours = Predict.GetMultipleContours(organsList,patient,path, modelTypeList = modelTypeList, thresholdList = thresList, withReal=withReal, tryLoad=tryLoad, save=save) 

    elif args.function == "BestThreshold":
        if len(organsList) > 1:
            print("\nThe best threshold can only be found for one organ at a time. Proceeding with the " + organsList[0])
        path = args.dataPath  
        modelType = args.modelType

        #if multiple model types were provided, chooses the first one
        if modelType is not None:
            if isinstance(modelType, list):
                modelType = modelType[0]
                if modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                    modelType = None
            elif modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                modelType = None

        if (modelType == None):
            while True:
                try:
                    modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                    modelType = str(modelType)
                    if " " in modelType:
                        modelType = modelType.split(" ")[0]
                        if modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                            break
                    elif modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                        break  
                except KeyboardInterrupt:
                    quit()
                except: pass  

        Test.BestThreshold(organsList[0], path, modelType, 500)

    elif args.function == "GetEvalData":
        if len(organsList) > 1:
            print("\nThe evaulation data can only be found for one organ at a time. Proceeding with the " + organsList[0])

        path = args.dataPath 
        if path == None:
            thresLoadPath = pathlib.Path(__file__).parent.absolute() 
        else:
            thresLoadPath = path

        modelType = args.modelType

        #if multiple model types were provided, chooses the first one
        if modelType is not None:
            if isinstance(modelType, list):
                modelType = modelType[0]
                if modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                    modelType = None
            elif modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                modelType = None

        if (modelType == None):
            while True:
                try:
                    modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                    modelType = str(modelType)
                    if " " in modelType:
                        modelType = modelType.split(" ")[0]
                        if modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                            break
                    elif modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                        break  
                except KeyboardInterrupt:
                    quit()
                except: pass  

        thres = args.thres
        if thres == None:
            try:  
                bestThresFile = open(str(os.path.join(thresLoadPath, "Models/Model_" + modelType.lower() + "_" + organsList[0].replace(" ", "")) + "_Thres.txt"), "r")   
                bestThres = bestThresFile.read()
                bestThresFile.close()
                thres = float(bestThres)
                print("\nBest threshold of " + str(thres) + " loaded for " + modelType + " " + organsList[0] + " predictions")
            except: 
                dataPath = 'Processed_Data/' + organsList[i] + "_Val/"
                dataFolder = os.path.join(thresLoadPath, dataPath)
                dataFiles = os.listdir(dataFolder)
                if len(dataFiles) > 0:
                    print("\nBest threshold has not been determined for " + modelType + " " + organsList[i] + " predictions. Launching the BestThreshold function.")
                    thres = Test.BestThreshold(organsList[i], path, modelType, 500)
                else: 
                    while True:
                        try:
                            thres = float(input("\nBest threshold could not be found for the " + organsList[i] + " " + modelType + " model. Please specify a threshold to use for plotting masks \n >"))
                            break    
                        except KeyboardInterrupt:
                            quit()
                        except: pass  
                        
        elif isinstance(thres, list):
            thres = float(thres[0])

        F_Score, recall, precision, accuracy, haussdorffDistance = Test.GetEvalData(organsList[0], path, thres, modelType)    
        print([F_Score, recall, precision, accuracy, haussdorffDistance])

    elif args.function == "PlotMasks":
        if len(organsList) > 1:
            print("\nMasks can only be plotted for one organ at a time. Proceeding with the " + organsList[0])

        path = args.dataPath 
        if path == None:
            thresLoadPath = pathlib.Path(__file__).parent.absolute() 
        else:
            thresLoadPath = path

        modelType = args.modelType

        #if multiple model types were provided, chooses the first one
        if modelType is not None:
            if isinstance(modelType, list):
                modelType = modelType[0]
                if modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                    modelType = None
            elif modelType.lower() != "unet" and modelType.lower() != "multiresunet": #make sure that the model is either unet or multiresunet
                modelType = None

        if (modelType == None):
            while True:
                try:
                    modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                    modelType = str(modelType)
                    if " " in modelType:
                        modelType = modelType.split(" ")[0]
                        if modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                            break
                    elif modelType.lower() == "unet" or modelType.lower() == "multiresunet":
                        break  
                except KeyboardInterrupt:
                    quit()
                except: pass  

        thres = args.thres
        if thres == None:
            try:  
                bestThresFile = open(str(os.path.join(thresLoadPath, "Models/Model_" + modelType.lower() + "_" + organsList[0].replace(" ", "")) + "_Thres.txt"), "r")   
                bestThres = bestThresFile.read()
                bestThresFile.close()
                thres = float(bestThres)
                print("\nBest threshold of " + str(thres) + " loaded for " + modelType + " " + organsList[0] + " predictions")
            except: 
                dataPath = 'Processed_Data/' + organsList[i] + "_Val/"
                dataFolder = os.path.join(thresLoadPath, dataPath)
                dataFiles = os.listdir(dataFolder)
                if len(dataFiles) > 0:
                    print("\nBest threshold has not been determined for " + modelType + " " + organsList[i] + " predictions. Launching the BestThreshold function.")
                    thres = Test.BestThreshold(organsList[i], path, modelType, 500)
                else: 
                    while True:
                        try:
                            thres = float(input("\nBest threshold could not be found for the " + organsList[i] + " " + modelType + " model. Please specify a threshold to use for plotting masks \n >"))
                            break    
                        except KeyboardInterrupt:
                            quit()
                        except: pass    
                        
        elif isinstance(thres, list):
            thres = float(thres[0])

        Test.TestPlot(organsList[0], path, modelType = modelType, threshold=thres)  



