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


def GetPatientContours(organ, patientFileName, threshold, withReal = True):
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
    try:
        contourImages, contours = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_contours.txt")),'rb'))      
    
    except:
        contours = []
        contourImages = []
        for CT in CTs:
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
            contours.append(contourPoints)
        contours = PostProcessing.FixContours(contours)    
        with open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_contours.txt")), "wb") as fp:
            pickle.dump([contourImages, contours], fp) 
            # plt.imshow(contourImage, cmap="gray")
            # plt.show()
    if withReal:
        existingContourImagesRaw= pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), str("Prediction_Patients/" + organ + "/" + patientFileName + "_ExistingContours.txt")), "rb"))
        existingContoursRaw = []
        for existingImage in existingContourImagesRaw:
            _, existingContourPoints = PostProcessing.MaskToContour(existingImage)   
            existingContoursRaw.append(existingContourPoints) 
        #fix the format of existingContoursRaw
        existingContours = []
        for plane in existingContoursRaw:
            planePoints = []
            if not len(plane) == 0:  
                for point in plane[0]:
                    planePoints.append(point[0].tolist())
            existingContours.append(planePoints)     
    #3d Render of predicted ROI:
        
    else:
        existingContours = []
    Test.PlotPatientContours(contours, existingContours)



    


