import numpy as np 
import matplotlib.pyplot as plt 
import Model
import torch
import torch.nn as nn
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
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))    
    model = model.cuda()      
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
            for CT in CTs:
                ipp = CT[1]
                zValues.append(float(ipp[2]))
                pixelSpacing = CT[2]
                sliceThickness= CT[3]
                x = torch.from_numpy(CT[0]).cuda()
                xLen, yLen = x.shape
                #need to reshape 
                x = torch.reshape(x, (1,1,xLen,yLen)).float()

                predictionRaw = (model(x)).cpu().detach().numpy()
                prediction = PostProcessing.Process(predictionRaw, threshold)
                # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
                # axs.imshow(prediction[0,0,:,:])
                # plt.show()
                contourImage, contourPoints = PostProcessing.MaskToContour(prediction[0,0,:,:])
                contourImages.append(contourImage)
                #contourPoints = PostProcessing.AddZToContour(contourPoints, ipp)
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
            x = torch.from_numpy(CT[0]).cuda()
            xLen, yLen = x.shape
            #need to reshape 
            x = torch.reshape(x, (1,1,xLen,yLen)).float()

            predictionRaw = (model(x)).cpu().detach().numpy()
            prediction = PostProcessing.Process(predictionRaw, threshold)
            # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
            # axs.imshow(prediction[0,0,:,:])
            # plt.show()
            contourImage, contourPoints = PostProcessing.MaskToContour(prediction[0,0,:,:])
            contourImages.append(contourImage)
            #contourPoints = PostProcessing.AddZToContour(contourPoints, ipp)
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



    


