import numpy as np 
import matplotlib.pyplot as plt 
import Model
import torch
import torch.nn as nn
import torch.onnx 
import pickle
import random
import os
import pathlib
import PostProcessing
import cv2 as cv
import DicomParsing
import Test

def GetContours(organ, patientName, path, threshold, modelType, withReal = True, tryLoad=True, plot=True):
    """Uses a pre trained model to predict contours for a given organ. Saves 
       the contours to the Predictions_Patients folder in a binary file. 

    Args:
        organ (str): the organ to predict contours for
        patientName (str): the name of the patient folder containing dicom 
            files (CT images) to predict contours for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model
        withReal (bool): True to get existing contours from dicom file, 
            defaults to True
        tryLoad (bool): True to try to load previously processed contours to 
            save time, defaults to True
        plot (bool) True to plot predicted contours, defaults to True

    Returns:
        contoursList (list): list of contour points from the 
            predicted contours
        existingContoursList (list): list of contour points from the 
            existing contours
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points [x,y,z]
            at a specific z value
        existingcontours (list): a list of lists. Each item is a list 
            of the existing contour points [x,y,z]
            at a specific z value

    """

    #with real loads pre=existing DICOM roi to compare the prediction with 
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for predicting: " + device.type)
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt")))  
    model = model.to(device)    
    model.eval()
    contoursList = [] #The 1d contours list to be returned
    existingContoursList = []
    #Make a list for all the contour images
    try: 
        CTs = pickle.load(open(os.path.join(path, str("Predictions_Patients/" + patientName + "_Processed.txt")), 'rb'))  
    except:

        CTs = DicomParsing.GetPredictionCTs(patientName, path)
    if tryLoad:
        try:
            contourImages, contours = pickle.load(open(os.path.join(path, str("Predictions_Patients/" + organ + "/" + patientName + "_predictedContours.txt")),'rb'))      
        except: 
            
            contours = []
            zValues = [] #keep track of z position to add to contours after
            contourImages = []

            exportNum = 0
            for CT in CTs:
                ipp = CT[1]
                zValues.append(float(ipp[2]))
                pixelSpacing = CT[2]
                sliceThickness= CT[3]
                x = torch.from_numpy(CT[0]).to(device)
                xLen, yLen = x.shape
                #need to reshape 
                x = torch.reshape(x, (1,1,xLen,yLen)).float()

                predictionRaw = (model(x)).cpu().detach().numpy()
                prediction = PostProcessing.Process(predictionRaw, threshold)
                contourImage, contourPoints = PostProcessing.MaskToContour(prediction[0,0,:,:])
                contourImages.append(contourImage)
                contours.append(contourPoints)  
       
            contours = PostProcessing.FixContours(contours)  
            contours = PostProcessing.AddZToContours(contours,zValues)                   
            contours = DicomParsing.PixelToContourCoordinates(contours, ipp, zValues, pixelSpacing, sliceThickness)
            contours = PostProcessing.InterpolateSlices(contours, patientName, organ, path, sliceThickness)

            for layer_idx in range(len(contours)):
                if len(contours[layer_idx]) > 0:
                    for point_idx in range(len(contours[layer_idx])):
                        x = contours[layer_idx][point_idx][0]
                        y = contours[layer_idx][point_idx][1]
                        z = contours[layer_idx][point_idx][2]
                        contoursList.append(x)
                        contoursList.append(y)
                        contoursList.append(z)
            with open(os.path.join(path, str("Predictions_Patients/" + organ + "/" + patientName + "_predictedContours.txt")), "wb") as fp:
                pickle.dump([contourImages, contours], fp)           
    else:
        contours = []
        zValues = [] #keep track of z position to add to contours after
        contourImages = []
        for CT in CTs:
            ipp = CT[1]
            zValues.append(float(ipp[2]))
            pixelSpacing = CT[2]
            sliceThickness= CT[3]
            x = torch.from_numpy(CT[0]).to(device)

            xLen, yLen = x.shape
            #need to reshape 
            x = torch.reshape(x, (1,1,xLen,yLen)).float()

            predictionRaw = (model(x)).cpu().detach().numpy()
            prediction = predictionRaw
            prediction[0,0,:,:] = PostProcessing.Process(predictionRaw[0,0,:,:], threshold)
            contourImage, contourPoints = PostProcessing.MaskToContour(prediction[0,0,:,:])
            contourImages.append(contourImage)
            contours.append(contourPoints)  
        
        contours = PostProcessing.FixContours(contours)  
        contours = PostProcessing.AddZToContours(contours,zValues)                   
        contours = DicomParsing.PixelToContourCoordinates(contours, ipp, zValues, pixelSpacing, sliceThickness)
        contours = PostProcessing.InterpolateSlices(contours, patientName, organ, path, sliceThickness)

        for layer_idx in range(len(contours)):
            if len(contours[layer_idx]) > 0:
                for point_idx in range(len(contours[layer_idx])):
                    x = contours[layer_idx][point_idx][0]
                    y = contours[layer_idx][point_idx][1]
                    z = contours[layer_idx][point_idx][2]
                    contoursList.append(x)
                    contoursList.append(y)
                    contoursList.append(z)

        with open(os.path.join(path, str("Predictions_Patients/" + organ + "/" + patientName + "_predictedContours.txt")), "wb") as fp:
            pickle.dump([contourImages, contours], fp)      

    existingContours = []
    
    if withReal:
        try:
            existingContours= pickle.load(open(os.path.join(path, str("Predictions_Patients/" + organ + "/" + patientName + "_ExistingContours.txt")), "rb"))  
            for layer_idx in range(len(existingContours)):
                if len(existingContours[layer_idx]) > 0:
                    for point_idx in range(len(existingContours[layer_idx])):
                        x = existingContours[layer_idx][point_idx][0]
                        y = existingContours[layer_idx][point_idx][1]
                        z = existingContours[layer_idx][point_idx][2]     
                        existingContoursList.append(x)
                        existingContoursList.append(y)
                        existingContoursList.append(z)
        except: 
            existingContours = DicomParsing.GetDICOMContours(patientName, organ, path)
            for layer_idx in range(len(existingContours)):
                if len(existingContours[layer_idx]) > 0:
                    for point_idx in range(len(existingContours[layer_idx])):
                        x = existingContours[layer_idx][point_idx][0]
                        y = existingContours[layer_idx][point_idx][1]
                        z = existingContours[layer_idx][point_idx][2]
                        existingContoursList.append(x)
                        existingContoursList.append(y)
                        existingContoursList.append(z)

    if plot==True:    
        Test.PlotPatientContours(contours, existingContours)
    return contoursList, existingContoursList, contours, existingContours     

def GetMultipleContours(organList, patientName, path, thresholdList, modelTypeList, withReal = True, tryLoad=True, plot=True): 
    """Calls the GetContours function to predict contours for each organ 
       in organList using a pretrained model and then plots all of the 
       predicted contours.

    Args:
        organList (list): a list of organs to predict contours for
        patientName (str): the name of the patient folder containing dicom 
            files (CT images) to predict contours for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        thresholdList (list): a list of floats containing the cutoff for deciding
            if a pixel is an organ (assigned a 1) or not (assigned a 0). 
            Corresponding thresholds must be in the same order as organList
        modelType (str): the type of model
        withReal (bool): True to get existing contours from dicom file, 
            defaults to True
        tryLoad (bool): True to try to load previously processed contours to 
            save time, defaults to True
        plot (bool) True to plot predicted contours, defaults to True

    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points [x,y,z]
            at a specific z value
        existingcontours (list): a list of lists. Each item is a list 
            of the existing contour points [x,y,z]
            at a specific z value

    """

    contours = []
    existingContours = []

    for i, organ in enumerate(organList): 
        print("\nPredicting contours for the " + organ + " with the threshold " + str(thresholdList[i]))
        combinedContours = GetContours(organ,patientName,path, modelType = modelTypeList[i], threshold = thresholdList[i], withReal=True, tryLoad=False, plot = False) 
        contours = contours + combinedContours[2]
        existingContours = existingContours + combinedContours[3]

    Test.PlotPatientContours(contours, existingContours)

    return contours, existingContours

def GetOriginalContours(organ, patientName, path):
    """Gets the original contours from a given patient's dicom file 
    for a given organ.

    Args:
        organ (str): the organ to get the original contours of
        patientName (str): the name of the patient folder containing dicom 
            files (CT images) to get contours from
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)

    Returns:
        existingcontours (list): a list of lists. Each item is a list 
            of the existing contour points [x,y,z]
            at a specific z value
        existingContoursList (list): a list of contour points from the 
            existing contours

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute() 

    existingContours = [] 
    existingContoursList = []

    try:
        existingContours= pickle.load(open(os.path.join(path, str("Predictions_Patients/" + organ + "/" + patientName + "_ExistingContours.txt")), "rb"))  
        for layer_idx in range(len(existingContours)):
            if len(existingContours[layer_idx]) > 0:
                for point_idx in range(len(existingContours[layer_idx])):
                    x = existingContours[layer_idx][point_idx][0]
                    y = existingContours[layer_idx][point_idx][1]
                    z = existingContours[layer_idx][point_idx][2]     
                    existingContoursList.append(x)
                    existingContoursList.append(y)
                    existingContoursList.append(z)
    except: 
        existingContours = DicomParsing.GetDICOMContours(patientName, organ, path)
        for layer_idx in range(len(existingContours)):
            if len(existingContours[layer_idx]) > 0:
                for point_idx in range(len(existingContours[layer_idx])):
                    x = existingContours[layer_idx][point_idx][0]
                    y = existingContours[layer_idx][point_idx][1]
                    z = existingContours[layer_idx][point_idx][2]
                    existingContoursList.append(x)
                    existingContoursList.append(y)
                    existingContoursList.append(z)

    return existingContours, existingContoursList






    


