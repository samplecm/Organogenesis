#!/usr/bin/env python

#local dependencies:
from numpy.core.fromnumeric import shape
import DicomParsing
import Train
import Test
import Predict
import PostProcessing
import re 
import argparse 

#Create a dictionary of organs and regular expressions for organs
organOps ={
    "Body": re.compile(r"body?"), 
    "Spinal Cord": re.compile(r"(spi?n?a?l)?(-|_| )?cord"),
    "Oral Cavity": re.compile(r"or?a?l(-|_| )cavi?t?y?"),
    "Left Parotid": re.compile(r"le?f?t?(-|_| )(par)o?t?i?d?"),
    "Right Parotid": re.compile(r"ri?g?h?t?(-|_| )(par)o?t?i?d?"),
    "Left Submandibular": re.compile(r"le?f?t(-|_| )subma?n?d?i?b?u?l?a?r?"),
    "Right Submandibular": re.compile(r"ri?g?h?t?(-|_| )subma?n?d?i?b?u?l?a?r?"),
    "Brain Stem": re.compile(r"b?r?a?i?n?(-|_| )stem"),
    "Larynx": re.compile(r"lary?n?(x|g?o?p?h?a?r?y?n?x?)?"),
    "All": re.compile(r"all")
}
#Create a list of possible functions
functionOps = [
    "Train",
    "GetContours",
    "BestThreshold",
    "GetEvalData",
    "PlotMasks",
    "FScore"
]




def main():
    print("Welcome to Organogenesis")
    print("------------------")

    #Keep a list of available structures for training/predicting
    OARs = ["Body", "Spinal Cord", "Oral Cavity", "Left Parotid", "Right Parotid", "All"] 

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
<<<<<<< HEAD
        Train.Train(OARs[chosenOAR], 7, 1e-3, path="/media/calebsample/Data/temp", processData=True, loadModel=False, preSorted=False)
        Test.Best_Threshold(OARs[chosenOAR],400)
=======
        Train.Train(chosenOARs[0], 35, 1e-3, path=None, processData=True, loadModel=False, preSorted=False, modelType = "MultiResUNet")
        #Test.Best_Threshold(OARs[chosenOAR],400)
>>>>>>> 72bc0a3d43d7245713fd7059d8401be4f1a9e761


        #Test.TestPlot(OARs[chosenOAR], path=None, threshold=0.1)  
    elif task == 2:    
        Predict.GetMultipleContours(chosenOARs,"P85",path = None, modelType = "multiresunet", thresholdList = [0.5], withReal=True, tryLoad=False) 
        
    elif task == 3:
<<<<<<< HEAD
        Test.Best_Threshold(OARs[chosenOAR], path=None, testSize=500, onlyMasks=False,onlyBackground=False)
=======
        Test.BestThreshold(chosenOARs[0], path=None, testSize=500, modelType = "UNet", onlyMasks=False, onlyBackground=False)
>>>>>>> 72bc0a3d43d7245713fd7059d8401be4f1a9e761
    elif task == 4:
        F_Score, recall, precision, accuracy, haussdorffDistance = Test.GetEvalData(chosenOARs[0], threshold=0.2, path = None, modelType = "unet")    
        print([F_Score, recall, precision, accuracy, haussdorffDistance])
        
    elif task == 5:
<<<<<<< HEAD
        array, y = Test.GetMasks(OARs[chosenOAR], "HN1046", path="/media/calebsample/Data/temp", threshold=0.1)
        import numpy as np
        print(np.amax(y))
        print(np.amax(array))
        Test.TestPlot(OARs[chosenOAR], path="/media/calebsample/Data/temp", threshold=0.1) 
=======
        #array, y = Test.GetMasks(OARs[chosenOAR], "P10", path=None, threshold=0.7, modelType = "UNet")
        #import numpy as np
        #print(np.amax(y))
        ##print(np.amax(array))
        #Test.TestPlot(chosenOARs[0], path=None, threshold=0.7, modelType = "UNet") 
        Test.PercentStats(chosenOARs[0], path = None)


>>>>>>> 72bc0a3d43d7245713fd7059d8401be4f1a9e761



if __name__ == "__main__":
    
        
    parser = argparse.ArgumentParser(
        description="Organogenesis: an open source program for autosegmentation of medical images"
    )
    parser.add_argument('-o', "--organs", help="Specify the organ that a model is to be trained to contour, or that a model is to be used for predicting/generating contours. \
        Please choose from: \n\n body,\n spinal-cord, \n oral-cavity, \n left-parotid, \n right-parotid, \n left-submandibular, \n right-submandibular, \n brain-stem, \n larynx/laryngopharynx", nargs = '+', default=None, type = str)
    parser.add_argument('-f', "--function", help = "Specify the function which is desired to be performed. Options include \"Train\": to train a model for predicting the specified organ, \
        \"GetContours\": Obtain predicted contour point clouds for a patient, \"BestThreshold\": find the best threhold for maximizing the models F score, \"FScore\": calculate the F score for the given organ's model, \
        \"PlotMasks\": Plot 2d CTs with both manually drawn and predicted masks for visual comparison", default=None, type=str)
    #Training parameters:    
    parser.add_argument("--lr", help="Specify the learning rate desired for model training.", default=None, type=float)
    parser.add_argument("--epochs", help="Specify the number of epochs to train the model for", default=None, type=int)
    parser.add_argument("--processData", help="True or False. True if patient DICOM data has already been processed into training/validation/test folders", default=False, action='store_true')
    parser.add_argument("--loadModel", help="True/False. True if a pre-existing model is to be loaded for continuing of training.", default=False, action='store_true')
    parser.add_argument("--dataPath", help="If data is not prepared in patient_files folder, specify the path to the directory containing all patient directories.",type=str, default=None)
    parser.add_argument("--preSorted", help="Specify whether or not patient data has already been sorted by \"good\" and \"bad\" contours", default=False, action='store_true')
    parser.add_argument("--modelType", help="Specify the model type. UNet or MultiResUNet", default=None, type=str)
    #GetContours parameters:
    parser.add_argument("--predictionPatientName", help= "Specify the name of the patient in the Predictions_Patient folder that you wish to predict contours for. Alternatively, supply the full path a patient's folder.",type=str, default=None)
    parser.add_argument("--thres", help="Specify the pixel mask threshold to use with the model", type=float, default=None, nargs = '+')
    parser.add_argument("--contoursWithReal", help="True/False. In GetContours, there is an option to plot predicted contours alongside the DICOM files manually contoured ones.", default=False , action='store_true')
    parser.add_argument("--loadContours", help="True/False. If the contours have already been created previously, tryLoad will attempt to load the processed data to save time.", default=False, action='store_true')
    


    args = parser.parse_args()
    v = vars(args)
    n_args = sum([ 1 for a in v.values( ) if a])
    if (n_args == 0):
        main()
    else:
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
                    organsSelected = input("\nInvalid or no organ(s) specified. Please specify the organ(s) that you wish to train or predict with separated by spaces.\n\nPlease choose from: \n\n body,\n spinal-cord, \n oral-cavity, \n left-parotid, \n right-parotid, \
                        \n left-submandibular, \n right-submandibular, \n brain-stem, \n larynx/laryngopharynx, \n all\n>")
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
                    functionSelection = input("\nInvalid function or no function specified. Please specify the function to be performed. Options include: \n\"Train\": to train a model for predicting the specified organ, \n\"GetContours\": Obtain predicted contour point clouds for a patient, \n\"BestThreshold\": find the best threhold for maximizing the model's F score, \n\"FScore\": calculate the F score for the given organ's model, \n\"PlotMasks\": plot 2d CTs with both manually drawn and predicted masks for visual comparison \n >")
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
            if (modelType == None):
                while True:
                    try:
                        modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                        modelType = str(modelType)
                        break    
                    except KeyboardInterrupt:
                        quit()
                    except: pass  

            processData = args.processData
            loadModel = args.loadModel
            dataPath = args.dataPath #If dataPath is None, then the program uses the data in the patient_files folder. If it is a path to a directory, data will be processed in this directory. 
            preSorted = args.preSorted

            Train.Train(organsList[0], numEpochs, lr, dataPath, processData, loadModel, preSorted, modelType)
            bestThreshold = Test.BestThreshold(organsList[0], dataPath, modelType = modelType, testSize = 400)

            Test.TestPlot(organsList[0], dataPath, threshold=bestThreshold, modelType = modelType)  

        elif args.function == "GetContours":
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
            thres = args.thres
            if (thres == None) or (len(thres) != len(organsList)):
                while True:
                    try:
                        thres = input("\nPlease specify the threshold to be used for contour prediction(s). If predicting with multiple organs, please enter the thresholds in the same order as the organs separated by a space\n >")      
                        thresList = list(thres.split(" "))
                        if len(thresList) == len(organsList):
                            break
                    except KeyboardInterrupt:
                        quit
                    except: pass    
            modelType = args.modelType
            if (modelType == None):
                while True:
                    try:
                        modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                        modelType = str(modelType)
                        break    
                    except KeyboardInterrupt:
                        quit()
                    except: pass     
            else: 
                thresList = thres
            for i, threshold in enumerate(thresList):
                thresList[i] = float(threshold)
            tryLoad = args.loadContours
            withReal = args.contoursWithReal   
            path = args.dataPath    
            
            combinedContours = Predict.GetMultipleContours(organsList,patient,path, modelType = modelType, thresholdList = thresList, withReal=True, tryLoad=False) 


        elif args.function == "BestThreshold":
            if len(organsList) > 1:
                print("\nThe best threshold can only be found for one organ at a time. Proceeding with the " + organsList[0])
            path = args.dataPath  
            modelType = args.modelType
            if (modelType == None):
                while True:
                    try:
                        modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                        modelType = str(modelType)
                        break    
                    except KeyboardInterrupt:
                        quit()
                    except: pass  
            Test.BestThreshold(organsList[0], path, modelType, 500)

        elif args.function == "FScore":
            if len(organsList) > 1:
                print("\nThe F score can only be found for one organ at a time. Proceeding with the " + organsList[0])
            thres = args.thres
            if thres == None:
                while True:
                    try:
                        thres = float(input("\nPlease specify the threshold to be used for contour prediction\n >"))       
                        break
                    except KeyboardInterrupt:
                        quit
                    except: pass    
            modelType = args.modelType
            if (modelType == None):
                while True:
                    try:
                        modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                        modelType = str(modelType)
                        break    
                    except KeyboardInterrupt:
                        quit()
                    except: pass  
            path = args.dataPath  
            F_Score, recall, precision, accuracy = Test.FScore(organsList[0], path, thres, modelType)    
            print([F_Score, recall, precision, accuracy])

        elif args.function == "PlotMasks":
            if len(organsList) > 1:
                print("\nMasks can only be plotted for one organ at a time. Proceeding with the " + organsList[0])
            thres = args.thres
            if thres == None:
                while True:
                    try:
                        thres = float(input("\nPlease specify the threshold to be used for contour prediction\n >"))       
                        break
                    except KeyboardInterrupt:
                        quit
                    except: pass 
            modelType = args.modelType
            if (modelType == None):
                while True:
                    try:
                        modelType = input("\nPlease specify the model type (UNet or MultiResUNet)\n >")
                        modelType = str(modelType)
                        break    
                    except KeyboardInterrupt:
                        quit()
                    except: pass  
            path = args.dataPath 
            Test.TestPlot(organsList[0], path, modelType = modelType, threshold=thres)  



