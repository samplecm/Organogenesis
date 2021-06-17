import numpy as np 
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
import math
import Model
import os
import pathlib
import random
import pickle
import Test


def Process(prediction, threshold):

    prediction = sigmoid(prediction)
    return prediction
    #prediction[0,1,:,:] = FilterBackground(prediction[0,1,:,:], threshold)
    prediction = FilterContour(prediction, threshold)
    
    return prediction


def sigmoid(z):
    return 1/(1 + np.exp(-(z)))      

def FilterContour(image, threshold):
    #this returns a binary prediction which is 1 in pixels that are above the threshold ,and zero otherwise
    xLen, yLen = image.shape
    #print(f"Contours: max pixel: {np.amax(image)}, min pixel: {np.amin(image)}")
    #return image #NormalizeImage(sigmoid(image))
    
    # image = (image > -2) #* image * (1 - (image < -5))
    # return image
    #image = NormalizeImage(image)
    filteredImage = np.zeros((xLen, yLen))
    for x in range(xLen):
        for y in range(yLen):
            if (image[x,y] > threshold):
                filteredImage[x,y] = 1
    return filteredImage   

def FilterBackground(image, threshold):
    #This is the opposite of filter contour, it makes background pixels (less than threshold) be 1 and mask pixels be 0. This creates a mask of the background
    xLen, yLen = image.shape
    #print(f"Background: max pixel: {np.amax(image)}, min pixel: {np.amin(image)}")
    #return image #nn.Softmax()(torch.from_numpy(image)).numpy()
    
    # image = (image < -1 ) * image

    #image = NormalizeImage(image)
    filteredImage = np.ones((xLen,yLen))
    for x in range(xLen):
        for y in range(yLen):
            if (image[x,y] < 1-threshold):
                filteredImage[x,y] = 0
    return filteredImage   

def NormalizeImage(image):
    #if its entirely ones or entirely zeros, can just return it as is. 
    if np.amin(image) == 1 and np.amax(image) == 1:
        return image
    elif np.amin(image) == 0 and np.amax(image) == 0:
        return image     
    ptp = np.ptp(image)
    amin = np.amin(image)
    return (image - amin) / ptp    

def MaskToContour(image):
    
    #forOpenCV's canny edge detection, define a maximum and minimum threshold value
    image = image.astype(np.uint8)
    edges = cv.Canny(image, 0,0.9)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #sometimes can return more than one contour list, so append these.
    if len(contours) == 0:
        return edges, contours
    elif not isinstance(contours[0], list):   
        return edges, contours
    else:    
        combinedContours = contours[0]
        for contour_idx in range(1, len(contours)):
            for point in contours[contour_idx]:
                combinedContours.append(point)
        return edges,  combinedContours        

def AddZToContours(contours, zValues):
    #currently contours is only x and y values, but need to add zValue to each point in each layer. 
    if len(contours) == 0:
        return contours
    for layer in range(len(contours)):    
        for point_idx in range(len(contours[layer])):
            contours[layer][point_idx] = [contours[layer][point_idx][0],contours[layer][point_idx][1],int(zValues[layer])]
        return contours


def GetMaskContour(image):
    #take an image of a mask and convert to a list of points describing the contour, as needed to construct a dicom file.
    image = image.astype(np.uint8)
    edges = cv.Canny(image, 0,0.9)

def FixContours(orig_contours):
    #orig_contours is currently in a ridiculous nested data structure. so fix it first. 
    contours = []
    for plane in orig_contours:
        planePoints = []
        if not len(plane) == 0:  
            for point in plane[0]:
                planePoints.append(point[0].tolist())
        contours.append(planePoints)    
    # print(orig_contours)
    # print(len(orig_contours))
    # print(len(orig_contours[25]))
    # print(orig_contours[25])
    # print(orig_contours[25][0][0])
    # print(orig_contours[25][0][0][0][0])
    # print(orig_contours)
    # print(len(contours))
    # print(len(contours[30]))
    # print(contours[30][0][0])

    #take in a list of all CT images, and check for slices which are false negatives. 
    
    #Make a list of all contour incices which have a contour on them:
    contourSlices = []
    idx = 0
    while idx < len(contours):
        if len(contours[idx]) > 0:
            contourSlices.append(idx)
        idx+=1    

    contourSlices.sort()
    contourSlices = FilterIslands(contourSlices)
    i = 0
    while i < len(contourSlices)-1:
        nextSlice_dist = contourSlices[i+1]-contourSlices[i]
        if nextSlice_dist > 1:
            try:
                contourSlices.insert(i+1, contourSlices[i] + 1)
                contours[contourSlices[i+1]] =  InterpolateContour(contours[i], contours[i+1], nextSlice_dist)
            except:
                pass    
        else:
            i+=1    
    
    contourSlices = []
    idx = 0
    while idx < len(contours):
        if len(contours[idx]) > 0:
            contourSlices.append(idx)
        idx+=1  
    contourSlices.sort()
    contourSlices = FilterIslands(contourSlices)
    for index in range(len(contours)):
        if index not in contourSlices:
            contours[index] = [] 
     #also, if there is 3 or less points on a plane with a slice in either direction with more than 8 and less than 3 slices away, interpolate
    i = 1
    while i < len(contourSlices)-1:
        if len(contours[contourSlices[i]]) <= 4 and len(contours[contourSlices[i]]) > 0:
            dist = 0 #distance to next slice on top with more than 8 points (if left at 0, just leave)
            try:
                if len(contours[contourSlices[i+1]])>8:
                    dist = 1
                elif len(contours[contourSlices[i+2]])>8:
                    dist = 2
                elif len(contours[contourSlices[i+3]])>8:
                    dist = 3
            except:
                pass     
            #make sure the slice below has more points, or else no point
            if dist != 0 and len(contours[contourSlices[i-1]])>len(contours[contourSlices[i]]):
                contours[contourSlices[i]] = InterpolateContour(contours[contourSlices[i-1]], contours[contourSlices[i + dist]], dist)
        i+=1
        

        
    #print(contourSlices)
    return contours
    

def InterpolateContour(contours1, contours2, distance):
    #idea is to take contours1, and create a new slice directly on top, which is an linear interpolation with the contour
    # at contours2, which is "distance" slices away from contours1
    newContour = []
    for point in contours1:
        point2 = ClosestPoint(point, contours2) #closest point in xy plane in contours2 list
        interpolatedPoint = InterpolatePoint(point, point2, distance) #linear interp between point1, point2
        newContour.append(interpolatedPoint)
    return newContour
    #for idx_1 in range(len(contours1)):

def ClosestPoint(point, contours):
    #find closest point to the given point within the contours list (in xy plane)
    minDist = 1000
    closestPoint = []
    x = point[0]
    y = point[1]
    for point2 in contours:
        x2 = point2[0]
        y2 = point2[1]
        diff = math.sqrt((x2-x)**2 + (y2-y)**2)

        if diff < minDist:
            closestPoint = point2 
            minDist = diff
    return closestPoint   
def InterpolatePoint(point1, point2, distance):
    #linear interpolation between point1 and point2, at a distance of 1 from point1 and distance-1 from point 2
    #first interpolate in x direction.
    
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]


    dx_dz = (x2-x1) / distance
    dy_dz = (y2-y1) / distance

    #get new point which is a distance of 1 away from point1
    x = x1 + dx_dz * 1
    y = y1 + dy_dz * 1

    return [x,y]


def FilterIslands(slices):
    #take the list of slices and remove islands by recursively taking the largest chunk in list when chunks separated by 5 or more slices
    maxGap, maxGapIndex = MaxGap(slices)
    #return largest half on either side of maxGap        
    if maxGap >= 5:
        if max(maxGapIndex,len(slices)-maxGapIndex) == maxGapIndex: #bottom half largest: 
            newSlices = slices[0:maxGapIndex]
        else:
            newSlices = slices[maxGapIndex:]

        return FilterIslands(newSlices)
    else:
        return slices  

def MaxGap(slices):
    #returns the largest separation of adjacent integers in a list of integers
    maxGap = 0
    maxGapIndex = 0 
    for i in range(1, len(slices)):
        sliceSeparation = slices[i]-slices[i-1]
        if sliceSeparation > maxGap:
            maxGap = sliceSeparation
            maxGapIndex = i
    return maxGap, maxGapIndex        


def Export_To_ONNX(organ):
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))      
    model.eval()

    #Now need a dummy image to predict with to save the weights
    x = np.zeros((512,512))
    x = torch.from_numpy(x)
    x = torch.reshape(x, (1,1,512,512)).float()
    torch.onnx.export(model,x,os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".onnx"), export_params=True, opset_version=10)



    try:
        session = onnxruntime.InferenceSession(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".onnx"))
    except (InvalidGraph, TypeError, RuntimeError) as e:
        raise e
def Infer_From_ONNX(organ, patient):    
    session = onnxruntime.InferenceSession(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".onnx"))
    inputName = session.get_inputs()[0].name 
    outputName = session.get_outputs()[0].name
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath)
    dataFiles = os.listdir(dataFolder)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)
    for d in filesRange:
        imagePath = dataFiles[d]
        #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
        data = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
        x = data[0, :, :]
        x = x.astype('float32')
        y = torch.from_numpy(data[1:2, :,:])
        xLen, yLen = x.shape
        #need to reshape 
        x = x.reshape((1,1,xLen,yLen))
        y = torch.reshape(y, (1,1,xLen,yLen)).float()   
        predictionRaw = session.run([outputName], {inputName: x})
        predictionRaw = np.reshape(np.array(predictionRaw), (512,512))
        prediction = np.reshape(Process(predictionRaw, 0.1), (1,1,512,512))
        y = y.numpy()
        maskOnImage = Test.MaskOnImage(x[0,0,:,:], prediction[0,0,:,:])
        ROIOnImage = Test.MaskOnImage(x[0,0,:,:], y[0,0,:,:])

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        
        #predictedContour = PostProcessing.MaskToContour(prediction[0,0,:,:])
        #contour = PostProcessing.MaskToContourImage(y[0,0,:,:])
        axs[0,0].imshow(ROIOnImage, cmap = "gray")
        axs[0,0].set_title("Original Image")
        #axs[0,1].hist(predictionRaw[0,0,:,:], bins=5)
        axs[0,1].imshow(maskOnImage, cmap = "gray")
        axs[0,1].set_title("Mask on Image")
        axs[1,0].imshow(y[0,0, :,:], cmap = "gray")
        axs[1,0].set_title("Original Mask")
        axs[1,1].imshow(prediction[0,0, :,:], cmap="gray")
        axs[1,1].set_title("Predicted Mask")

        
        # axs[2,0].imshow(y[0,1, :,:], cmap = "gray")
        # axs[2,0].set_title("Original Background")
        # axs[2,1].imshow(prediction[0,1, :,:], cmap = "gray")
        # axs[2,1].set_title("Predicted Background")
        plt.show()
        




