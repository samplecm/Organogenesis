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



def GetContours(organ, patientFileName, threshold, withReal = True, tryLoad=True):
    #with real loads pre=existing DICOM roi to compare the prediction with 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device being used for training: " + device.type)
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))    
    model = model.to(device)    
    model.eval()

    #Make a list for all the contour images
    try: 
        CTs = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + patientFileName + "_Processed.txt")), 'rb'))  
    except:

        CTs = DicomParsing.GetPredictionData(patientFileName,organ)
    if tryLoad:
        try:
            contourImages, contours = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_predictedContours.txt")),'rb'))      
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
            with open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_predictedContours.txt")), "wb") as fp:
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
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_predictedContours.txt")), "wb") as fp:
            pickle.dump([contourImages, contours], fp)        
    existingContours = []
    if withReal:
        try:
            existingContours= pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_ExistingContours.txt")), "rb"))       
        except: 
            pass
    Test.PlotPatientContours(contours, existingContours)



    


