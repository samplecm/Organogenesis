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
    "Spinal Cord": re.compile(r"(spin?a?l?)?(-,_, )?cord"),
    "Oral Cavity": re.compile(r"or?a?l?cavi?t?y?"),
    "Left Parotid": re.compile(r"le?f?t?(-|_| )(par)o?t?i?d?"),
    "Right Parotid": re.compile(r"ri?g?h?t?(-|_| )(par)o?t?i?d?"),
    "Left Submandibular": re.compile(r"le?f?t(-,_, )subma?n?d?i?b?u?l?a?r?"),
    "Right Submandibular": re.compile(r"ri?g?h?t?(-,_, )subma?n?d?i?b?u?l?a?r?"),
    "Brain Stem": re.compile(r"b?r?a?i?n?(-,_, )stem"),
    "Larynx": re.compile(r"lary?n?(x|g?o?p?h?a?r?y?n?x?)?") 
}
#Create a list of possible functions
functionOps = [
    "Train",
    "GetContours",
    "BestThreshold",
    "GetEvalData",
    "PlotMasks"
]


def main():
    print("Welcome to Organogenesis")
    print("------------------")

    #Keep a list of available structures for training/predicting
    OARs = ["Body", "Spinal Cord", "Oral Cavity", "Left Parotid", "Right Parotid"] #Later will add an "all" option

    #Need to get user input. Make a string to easily ask for a number corresponding to an OAR.
    ChooseOAR_string = "Please enter the number for the organ you wish to contour / train a model for \n>>"
    for i in range(len(OARs)):
        ChooseOAR_string += str(i + 1) + ": " + OARs[i] + "\n" #create list of options
    
    while True: #wait for user input    
        try:
            chosenOAR = int(input(ChooseOAR_string)) - 1
            if chosenOAR < len(OARs):
                break
        except KeyboardInterrupt:
            quit()    
        except: pass     

    #Now determine if the goal is to train or to find contours, etc
    chooseTask_string = "Please enter the number for the desired task\n"
    chooseTask_string += "1. Train a UNet model for predicting " + str(OARs[chosenOAR])
    chooseTask_string += "\n2. Predict " + str(OARs[chosenOAR]) + " contours using an existing model"
    chooseTask_string += "\n3. Determine threshold accuracies for predictions of the " + str(OARs[chosenOAR])
    chooseTask_string += "\n4. Determine the evaulation data (F score and 95th percentile Haussdorf distance) for the validation set of the " + str(OARs[chosenOAR])
    chooseTask_string += "\n5. Plot predicted masks for the  " + str(OARs[chosenOAR])
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
        Train.Train(OARs[chosenOAR], 7, 1e-3, path=None, processData=False, loadModel=False, preSorted=True, modelType = "MultiResUNet")
        Test.Best_Threshold(OARs[chosenOAR],400)

        Test.TestPlot(OARs[chosenOAR], path=None, threshold=0.1)  
    elif task == 2:    
        contoursList, existingContoursList = Predict.GetContours(OARs[chosenOAR],"P85", path=None, threshold = 0.15, withReal=True, tryLoad=False) 
        
    elif task == 3:
        Test.BestThreshold(OARs[chosenOAR], path=None, testSize=500, onlyMasks=False,onlyBackground=False)
    elif task == 4:
        F_Score, recall, precision, accuracy = Test.FScore(OARs[chosenOAR], threshold=0.2, path = None)    
        print([F_Score, recall, precision, accuracy])
    elif task == 5:
        array, y = Test.GetMasks(OARs[chosenOAR], "P2", path=None, threshold=0.7)
        import numpy as np
        print(np.amax(y))
        print(np.amax(array))
        #Test.TestPlot(OARs[chosenOAR], path="/media/calebsample/Data/temp", threshold=0.1) 


   

if __name__ == "__main__":
    
        
    parser = argparse.ArgumentParser(
        description="Organogenesis: an open source program for autosegmentation of medical images"
    )
    parser.add_argument('-o', "--organ", help="Specify the organ that a model is to be trained to contour, or that a model is to be used for predicting/generating contours. \
        Please choose from: \n\n body,\n spinal-cord, \n oral-cavity, \n left-parotid, \n right-parotid, \n left-submandibular, \n right-submandibular, \n brain-stem, \n larynx/laryngopharynx", default=None, type=str)
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
    #GetContours parameters:
    parser.add_argument("--predictionPatientName", help= "Specify the name of the patient in the Predictions_Patient folder that you wish to predict contours for. Alternatively, supply the full path a patient's folder.",type=str, default=None)
    parser.add_argument("--thres", help="Specify the pixel mask threshold to use with the model", type=float, default=None)
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
        organ = ""

        organMatch = False
        for key in organOps:
            if (re.match(organOps[key], args.organ)): 
                organMatch = True
                organ = key 
                print("Selected Organ:" + key)

        if (args.organ == None) or (organMatch==False):
            while True: #get user input
                organMatch = False 
                try:
                    organSelection = input("\nPlease specify a proper organ that you wish to train or use a model for predicting. \n\nPlease choose from: \n\n body,\n spinal-cord, \n oral-cavity, \n left-parotid, \n right-parotid, \
                        \n left-submandibular, \n right-submandibular, \n brain-stem, \n larynx/laryngopharynx \n >")
                    for key in organOps:
                        if (re.match(organOps[key], organSelection.lower())):
                            print("Selected Organ:" + key)
                            organMatch = True
                            organ = key
                            break
                    if organMatch:
                        break    
                except KeyboardInterrupt:
                    quit()
                except: pass  

        #Now ensure that a proper function has been requested         
        functionMatch = False 
        for function in functionOps:
            if function == args.function:
                functionMatch = True 
                break
        if not functionMatch:
            while True: #get user input 
                try:
                    functionSelection = input("\nPlease specify the function to be performed. Options include \"Train\": to train a model for predicting the specified organ, \
            \"GetContours\": Obtain predicted contour point clouds for a patient, \"BestThreshold\": find the best threhold for maximizing the models F score, \"FScore\": calculate the F score for the given organ's model, \
            \"PlotMasks\": Plot 2d CTs with both manually drawn and predicted masks for visual comparison \n >")
                    for function in functionOps:
                        if (function == functionSelection):
                            print("Selected:" + functionSelection)
                            functionMatch = True
                            organ = key
                            break
                    if functionMatch:
                        break    
                except KeyboardInterrupt:
                    quit()
                except: pass  
        else: 
            print("Selected:" + args.function)        


        #Now perform the specified function:
        if args.function == "Train":

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

            processData = args.processData
            loadModel = args.loadModel
            dataPath = args.dataPath #If dataPath is None, then the program uses the data in the patient_files folder. If it is a path to a directory, data will be processed in this directory. 
            preSorted = args.preSorted

            Train.Train(organ, numEpochs, lr, dataPath, processData, loadModel, preSorted)
            bestThreshold = Test.BestThreshold(organ, dataPath, 400)

            Test.TestPlot(organ, dataPath, threshold=bestThreshold)  

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
            if thres == None:
                while True:
                    try:
                        thres = float(input("Please specify the threshold to be used for contour prediction\n >"))       
                        break
                    except KeyboardInterrupt:
                        quit
                    except: pass     
            tryLoad = args.loadContours
            withReal = args.contoursWithReal   
            path = args.dataPath     
            Predict.GetContours(organ ,patient,path, threshold = 0.15, withReal=True, tryLoad=False) 

        elif args.function == "BestThreshold":
            path = args.dataPath     
            Test.BestThreshold(organ, path, 500)

        elif args.function == "FScore":
            thres = args.thres
            if thres == None:
                while True:
                    try:
                        thres = float(input("Please specify the threshold to be used for contour prediction\n >"))       
                        break
                    except KeyboardInterrupt:
                        quit
                    except: pass     
            path = args.dataPath            
            F_Score, recall, precision, accuracy = Test.FScore(organ, path, thres)    
            print([F_Score, recall, precision, accuracy])

        elif args.function == "PlotMasks":
            thres = args.thres
            if thres == None:
                while True:
                    try:
                        thres = float(input("Please specify the threshold to be used for contour prediction\n >"))       
                        break
                    except KeyboardInterrupt:
                        quit
                    except: pass     
            path = args.dataPath 
            Test.TestPlot(organ, path, threshold=thres)  





