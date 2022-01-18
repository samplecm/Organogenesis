import numpy as np
from numpy.core.fromnumeric import nonzero 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, dataset 
import cv2 as cv
import matplotlib.pyplot as plt
import math
import Model
from Dataset import CTDataset
import os
import pathlib
import random
import pickle
import Test
import DicomParsing
from scipy.spatial import ConvexHull
import copy
import statistics


def ThresholdRescaler(organ, modelType, path):
    #This function goes through the validation set for a specific model and rescales data so that the median prediction value for within a training mask is
    #0 (before the sigmoid or other processing).
    #In other words, it implements an intercept to go along with predictions to get a more stable threshold.
    if path==None: #if a path to data was not supplied, assume that patient data has been placed in the Patient_Files folder in the current directory. 
        path = pathlib.Path(__file__).parent.absolute() 
    maskPredictions = []
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt")))  
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))) 
    #model.eval()
    #go through every file in validation set and look for mask images
    validationPath = os.path.join(path, "Processed_Data", str(organ +"_Val"))
    dataFiles = os.listdir(validationPath)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Determining rescaling factor for ' + organ + " model")

    numFiles = len(dataFiles)
    numCompleted = 0
    for image in dataFiles:

        if numCompleted % 10 == 0:
            print("Finished " + str(numCompleted) + "/" + str(numFiles))
        numCompleted = numCompleted + 1
        data = pickle.load(open(os.path.join(validationPath, image), 'rb'))
        # if np.amax(data[0][1,:,:] != 0): #if there is no mask on this image         
        #     continue
        if np.amax(data[0][1,:,:]) == 0: #if there is no mask on this image         
            continue

        x = torch.from_numpy(data[0][0, :, :])
        y = data[0][1,:,:]
        x = x.to(device)

        xLen, yLen = x.shape
        #need to reshape 
        x = torch.reshape(x, (1,1,xLen,yLen)).float()
        prediction = (model(x)).cpu().detach().numpy()
        #print(np.amax(prediction[0,0,:,:]))
        # plt.imshow(data[0][0,:,:])
        # #plt.imshow(prediction[0,0,:,:])
        # plt.show()
        #filter out any pixels that aren't in the mask
        maskPixels = y * prediction[0,0,:,:]
        nonZeroVals = maskPixels[np.nonzero(maskPixels)]
        for val in nonZeroVals:
            maskPredictions.append(val)
        
    medianVal = np.median(maskPredictions)
    std = np.std(maskPredictions)

    #Now the intercept is the difference between the 0 and the statistic value.
    intercept = - medianVal  + std/2
    print("Calculated intercept for model to be " + str(intercept))
    saveFileName = modelType.lower() + "_" + organ.replace(" ", "") + "_rescale_intercept.txt" 
    with open(os.path.join(path, "Models/ScalingFactors", saveFileName ), 'wb') as pick:
        pickle.dump(intercept, pick)    
    print("Finished.")    
    return intercept






def Process(prediction, threshold, modelType, organ):
    """Performs a sigmoid and filters the prediction to a given threshold.

    Args:
        prediction (2D numpy array): an array of the predicted pixel values 
        threshold (float) : the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)

    """
    
    #first apply the rescaling intercept
    saveFileName = modelType.lower() + "_" + organ.replace(" ", "") + "_rescale_intercept.txt" 
    try:
        with open(os.path.join(os.getcwd(), "Models/ScalingFactors", saveFileName ), 'rb') as fp:
            intercept = pickle.load(fp)
    except: 
        intercept = ThresholdRescaler(organ, modelType, path=None)
        with open(os.path.join(os.getcwd(), "Models/ScalingFactors", saveFileName ), 'wb') as fp:
            pickle.dump(intercept, fp)


    prediction += intercept    
    prediction = sigmoid(prediction)
    
    prediction = FilterContour(prediction, threshold)
    
    return prediction

def AddInterpolatedPoints(orig_contours):
    """This makes sure that each slice of contours has at least 30 points
    Args: 
        contours (list): the contour list for a single patient
    """
    contours = copy.deepcopy(orig_contours)
    #Now add in sufficient number of points 
    for contour_idx in range(len(contours)): 
        contour = contours[contour_idx]
        numPointsOrig = len(contour)
        numPoints = len(contour)
        if numPoints > 40:
            continue
        if numPoints < 3:
            contours[contour_idx] = []
            continue


        pointIncreaseFactor = 1
        while numPoints < 40:  
            numPoints = numPoints + numPointsOrig
            pointIncreaseFactor = pointIncreaseFactor + 1      
        increasedContour = []
        for point_idx in range(len(contour)-1):    
            increasedContour.append(contour[point_idx].copy())
            for extraPointNum in range(pointIncreaseFactor):
                scaleFactor = extraPointNum / (pointIncreaseFactor + 1)
                newX = (contour[point_idx+1][0] - contour[point_idx][0]) * scaleFactor + contour[point_idx][0]
                newY = (contour[point_idx+1][1] - contour[point_idx][1]) * scaleFactor + contour[point_idx][1]
                z = contour[point_idx][2]
                newPoint = [newX, newY, z]
                increasedContour.append(newPoint)
        #Now do it for the last point connected to the first
        for extraPointNum in range(pointIncreaseFactor):
            scaleFactor = extraPointNum / (pointIncreaseFactor + 1)
            newX = (contour[0][0] - contour[-1][0]) * scaleFactor + contour[-1][0]
            newY = (contour[0][1] - contour[-1][1]) * scaleFactor + contour[-1][1]
            z = contour[-1][2]
            newPoint = [newX, newY, z]
            increasedContour.append(newPoint)        
        contours[contour_idx] = increasedContour

    return contours 

def sigmoid(z):
    """Performs a sigmoid function on z. 

    Args:
        z (ndarray): the array to be modified 

    Returns:
        ndarray: the array after a sigmoid function has been applied

    """
    return 1/(1 + np.exp(-(z)))      

def FilterContour(image, threshold):
    """Creates a binary mask. Pixels above the threshold are 
       assigned a 1 (are the organ), and 0 (are not the organ) otherwise.
       This is the opposite of FilterBackground.

    Args:
        image (2D numpy array): an array of pixels with values between 0 and 1
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)

    Returns:
        filteredImage (2D numpy array): the binary mask

    """
    return np.where(image > threshold, 1, 0)

def FilterBackground(image, threshold):
    """Creates a binary mask of the background of the image. Pixels below the 
       threshold are assigned a 1 (are not the organ), and 0 (are the organ) otherwise.
       This is the opposite of FilterContour.

    Args:
        image (2D numpy array): an array of pixels with values between 0 and 1
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)

    Returns:
        filteredImage (2D numpy array): the binary mask of the background

    """

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
    """Normalizes an image between 0 and 1.  

    Args:
        image (ndarray): the image to be normalized 

    Returns:
        ndarray: the normalized image

    """

    #if its entirely ones or entirely zeros, can just return it as is. 
    if np.amin(image) == 1 and np.amax(image) == 1:
        return image
    elif np.amin(image) == 0 and np.amax(image) == 0:
        return image     
    ptp = np.ptp(image)
    amin = np.amin(image)
    return (image - amin) / ptp    

def MaskToContour(image):
    """Creates a contour from a mask.  

    Args:
        image (2D numpy array): the mask to create a contour from

    Returns:
        edges (2D array): an array of the same dimensions as image 
            with values 0 (pixel is not an edge) or 255 (pixel is 
            an edge)
        contours (ndarray): an array of the contour points for image
        combinedContours (ndarray): an array of the contour points 
            for image combined from multiple contours


    """
    
    #forOpenCV's canny edge detection, define a maximum and minimum threshold value
    image = image.astype(np.uint8)
    edges = cv.Canny(image, 0,0.7)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    #sometimes can return more than one contour list, so append these.
    if len(contours) == 0:
        return contours
    elif not isinstance(contours[0], list):   
        return contours
    else:    
        combinedContours = contours[0] #now add points for all different contours which may exist
        for contour_idx in range(1, len(contours)):
            for point in contours[contour_idx]:
                combinedContours.append(point)
        return combinedContours        

def AddZToContours(contours, zValues):
    """Adds the z value to each point in contours as it only 
       contains x and y values.   

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (only x and y values) 
            at a specific z value
    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value

    """

    if len(contours) == 0:
        return contours
    for layer in range(len(contours)):    
        for point_idx in range(len(contours[layer])):
            contours[layer][point_idx] = [contours[layer][point_idx][0],contours[layer][point_idx][1],int(zValues[layer])]
            contours[layer][point_idx] = [contours[layer][point_idx][0],contours[layer][point_idx][1],int(zValues[layer])]
    return contours

class ContourPredictionError(Exception):
    """Exception raised if there are no contour points in a contour. 

    """

    def __init__(self, message = "Contour Prediction Error. No contour points were predicted by the model. Please check that you are using the right threshold and try again."):
        """Instantiation method of ContourPredictionError. 

        Args: 
            message (string): the message to be printed if the contour 
                prediction error is raised 

        """
        self.message = message
        super().__init__(self.message)

def FixContours(orig_contours):
    """Creates additional interpolated contour slices if there 
       are missing slice or if the slice has less than 4 points.

    Args:
        orig_contours (list): a list of lists. Each item is a list 
            of the predicted contour points as 1D arrays [x,y]
            at a specific z value
    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points [x,y]
            at a specific z value 

    """

    #raise ContourPredictionError if no contour points are predicted
    maxLength = 0
    for slice in orig_contours: 
        if len(slice) > maxLength:
            maxLength = len(slice)

    if maxLength == 0:
        raise ContourPredictionError

    #orig_contours is currently in a ridiculous nested data structure. so fix it first. 
    contours = []
    for plane in orig_contours:
        planePoints = []
        if not len(plane) == 0:  
            for point in plane[0]:
                planePoints.append(point[0].tolist())
        contours.append(planePoints)    

    #take in a list of all CT images, and check for slices which are false negatives. 
    
    #Make a list of all contour incices which have a contour on them:
    contourSlices = []
    idx = 0
    while idx < len(contours):
        contour = contours[idx]
        if len(contours[idx]) > 0:
            contourSlices.append(idx)
        idx+=1    

    contourSlices.sort()
    contourSlices = FilterIslands(contourSlices)
    # i = 0
    # while i < len(contourSlices)-1:
    #     nextSlice_dist = contourSlices[i+1]-contourSlices[i]
    #     if nextSlice_dist > 1:
    #         try:
    #             contours[contourSlices[i]+1] =  InterpolateContour(contours[contourSlices[i]], contours[contourSlices[i+1]], nextSlice_dist)
    #             contourSlices.insert(i+1, contourSlices[i] + 1)
    #         except:
    #             pass    
    #     else:
    #         i+=1    
    
    contours = ValidateSlices(contours)
    #  #also, if there is 3 or less points on a plane with a slice in either direction with more than 8 and less than 3 slices away, interpolate
    
    # i = 0
    # while i < len(contourSlices)-1:
    #     if len(contours[contourSlices[i]]) <= 4:
    #         dist = 0 #distance to next slice on top with more than 8 points (if left at 0, just leave)
    #         try:
    #             if len(contours[contourSlices[i+1]])>8:
    #                 dist = 1
    #             elif len(contours[contourSlices[i+2]])>8:
    #                 dist = 2
    #             elif len(contours[contourSlices[i+3]])>8:
    #                 dist = 3
    #         except:
    #             pass     
    #         #make sure the slice below has more points, or else no point
    #         try:
    #             if dist != 0 and len(contours[contourSlices[i-1]])>8:
    #                 contours[contourSlices[i]] = InterpolateContour(contours[contourSlices[i-1]], contours[contourSlices[i + dist]], dist+1)
    #             elif dist != 0 and len(contours[contourSlices[i-2]])>8:
    #                 contours[contourSlices[i]] = InterpolateContour(contours[contourSlices[i-2]], contours[contourSlices[i + dist]], dist+2)   
    #         except:
    #             print("Error interpolating in FixContours function.")
    #             exit()         
    #     i+=1
        

        
    #print(contourSlices)
    return contours
    

def InterpolateContour(contour1, contour2, totalDistance, dist=1):
    """Creates a linearly interpolated contour slice between contours1 and contours2. 

    Args:
        contours1 (list): the first slice to be interpolated with. 
            A list of [x,y] coordinates
        contours2 (list): the second slice to be interpolated with
            A list of [x,y] coordinates
        distance (int): the difference in slice indices between contours1 and contours2

    Returns:
        newContour (list): the interpolated slice. A list of [x,y] coordinates
    """
    #want to interpolate from the contour with the greatest area (approximate as one with most points)
    if len(contour2) > len(contour1):
        from_contour = contour2
        to_contour = contour1
        dist = totalDistance-dist
    else:
        from_contour = contour1
        to_contour = contour2    

    newContour = []
    for point in from_contour:
        point2 = ClosestPoint(point, to_contour) #closest point in xy plane in contours2 list
        interpolatedPoint = InterpolatePoint(point, point2, totalDistance, distance=dist) #linear interp between point1, point2
        newContour.append(interpolatedPoint)
    return newContour
    #for idx_1 in range(len(contours1)):

def ClosestPoint(point, contours):
    """Finds the closest point to the given point within the
       contours list.

    Args:
        point (list): the point [x,y]
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (only x and y values) 
            at a specific z value

    Returns:
        closestPoint (list): the closest point to the given point 
             in the contours list [x,y]
    """

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

def InterpolatePoint(point1, point2, totalDistance, distance=1):
    """Perfoms linear interpolation between point1 and point2.

    Args:
        point1 (list): the first point to be interpolated [x,y]
        point2 (list): the second point to be interpolated [x,y]
        totalDistance (int, float): the z distance between point 1 and point 2
        distance (int, float): the z distance between point 1 and the z value 
            of the interpolated point

    Returns:
        list: the interpolated point [x,y]
    """

    #first interpolate in x direction.
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]


    dx_dz = (x2-x1) / float(totalDistance)
    dy_dz = (y2-y1) / float(totalDistance)

    #get new point which is a distance of 1 away from point1
    x = x1 + dx_dz * float(distance)
    y = y1 + dy_dz * float(distance)

    return [x,y]

def ValidateSlices(contours):
    """Checks the area of contour slices, and interpolates when abnormally small slices are surrounded by two larger areas,

    Args:
      contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value

    """
    areas = [] #list for all areas for slices
    
    for slice in contours:
        areas.append(Area_of_Slice(slice))  
    areas_end = areas[338:len(areas)-1]  
    median_area = statistics.median(filter(lambda area: area > 0, areas))
    cutoff_area = 0.05 * median_area
    #first look for slices with <5% median area which have one below and on top which are greater than 0 
    slice_below=False
    last_good_slice = None #keep track of index of last good slice   
    
    for i, slice in enumerate(contours):
        area = areas[i]
        if area > cutoff_area:
            slice_below=True
            last_good_slice=i

        elif slice_below==True: #inadequate area
            next_good_slice = None #look for next slice with area > 5%
            for j in range(i, min(i+4, len(contours))):
                if areas[j] > cutoff_area:
                    next_good_slice = j
                    break
            if next_good_slice != None:
                slices_separation = next_good_slice - last_good_slice
                new_contour = InterpolateContour(contours[last_good_slice], contours[next_good_slice], slices_separation, dist=i-last_good_slice)
                new_area = Area_of_Slice(new_contour)
                contours[i] = new_contour
                areas[i] = new_area
                areas_end = areas[338:len(areas)-1]  
                #last_good_slice = i
            else:
                #if can't interpolate to a good slice, delete contour
                contours[i] = []   
        else:
            contours[i] = []         

    #now do another loop through, and if any given slice is less than 30% of area directly on top and below it, interpolate.
    for i in range(1, len(contours)-1):
        area = areas[i]
        if area != 0: 
            if areas[i-1] > area / 0.3 and areas[i+1] > area / 0.3:
                new_contour = InterpolateContour(contours[i-1], contours[i+1], 2, dist=1)
                contours[i] = new_contour
            elif areas[i-1] * 1.5 < area  and areas[i+1] * 1.5 < area:
                new_contour = InterpolateContour(contours[i-1], contours[i+1], 2, dist=1)
                contours[i] = new_contour    
                
    return contours            



            



    print("")

def Area_of_Slice(slice):
    area = 0
    j = len(slice)-1
    for i in range(j):
        area += (slice[j][0]+slice[i][0])*(slice[j][1] - slice[i][1])
        j = i
    return abs(area) / 2   


def FilterIslands(slices):
    """Removes islands by recursively taking the largest chunk in the list. 
        Lists are separated by 5 or more slices. 

    Args:
        slices (list): list of indices for slices with at least one point
    Returns:
        slices (list): the list of indices with islands removed 

    """

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
    """Finds the largest separation of adjacent integers 
    in a list of integers.   

    Args:
        slices (list): list of indices for slices with at least one point
    Returns:
        maxGap (int): the largest difference between adjacent indices in slices
        maxGapIndex (int): the index at which the largest gap occurs 

    """

    maxGap = 0
    maxGapIndex = 0 
    for i in range(1, len(slices)):
        sliceSeparation = slices[i]-slices[i-1]
        if sliceSeparation > maxGap:
            maxGap = sliceSeparation
            maxGapIndex = i
    return maxGap, maxGapIndex        


def Export_To_ONNX(organ):
    """Takes a PyTorch model and converts it to an Open Neural Network Exchange 
       (ONNX) model for operability with other programming languages. The new 
       model is saved into the Models folder.

    Args:
        organ (str): the name of the organ for which a PyTorch model is to 
            be converted to an ONNX model
        modelType (str): specifies whether a UNet or MultiResUnet model is 
            to be converted

    """
    model = Model.UNet()
    #if the model is not UNet switch to MultiResUNet
    try: 
        model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + organ.replace(" ", "") + ".pt")))  
    except Exception as e: 
        if "Missing key(s) in state_dict:" in str(e): #check if it is the missing keys error
            model = Model.MultiResUNet()
            model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + organ.replace(" ", "") + ".pt")))  
        else: 
            print(e)
            os._exit(0)    
    #model.eval()

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
    """Predicts organ masks using an ONNX model and plots slices for viewing. 
       The ONNX model must be in the Models directory with the name 
       “Model_{organ}.” This function is limited to viewing and cannot be 
       used for creating contour lists or DICOM files.

    Args:
      organ (str): organ for which binary masks are to be predicted 
        using an ONNX model
      patient (str): the name of the patient folder within the Patient_Files 
        directory which is to have organ masks predicted

"""
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


def InterpolateSlices(contours, patientName, organ, path, sliceThickness):
    """Interpolates slices with an unreasonable area, an unreasonable number of 
       points, or with missing slices using the closest slice above and below.

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value
        patientName (str): the name of the patient folder to 
            interpolate contours for
        organ (str): the organ to interpolate contours for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        sliceThickness (float): the distance between each CT slice in mm

    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted and interpolated contour points 
            (x, y, and z values) at a specific z value

    """

    slicesToFix = list(set(UnreasonableArea(contours, organ, path) + MissingSlices(contours, sliceThickness) + UnreasonableNumPoints(contours, organ, path)))

    contourZValues = GetZValues(contours)
    maxZValue = max(contourZValues)
    minZValue = min(contourZValues)

    for zValue in slicesToFix:
        #find the z values of the closest slices with reasonable areas 
        zValueAbove = zValue
        zValueBelow = zValue 
        i = 1
        while zValueBelow == zValue: 
            if round(zValue - i*sliceThickness,2) not in slicesToFix:
                zValueBelow = zValue - i*sliceThickness
                distanceBelow = i*sliceThickness
            i += 1

        i = 1
        while zValueAbove == zValue: 
            if round(zValue + i*sliceThickness,2) not in slicesToFix:
                zValueAbove = zValue + i*sliceThickness
                distanceAbove = i*sliceThickness 
            i += 1

        #don't interpolate if the closest slices were out of the z value range
        if zValueAbove > maxZValue or zValueBelow < minZValue: 
            continue

        pointsListAbove = GetPointsAtZValue(contours, zValueAbove)
        pointsListBelow = GetPointsAtZValue(contours, zValueBelow)

        interpolatedPointsList = []

        if len(pointsListAbove) > len(pointsListBelow): #choose the slice with the largest number of contour points
            for pointAbove in pointsListAbove:
                pointBelow = ClosestPoint(pointAbove, pointsListBelow) 
                interpolatedPoint = InterpolatePoint(pointAbove, pointBelow, distanceBelow+distanceAbove, distanceAbove) 
                interpolatedPointsList.append([round(interpolatedPoint[0],2), round(interpolatedPoint[1],2), round(zValue,2)])
        else: 
            for pointBelow in pointsListBelow:
                pointAbove = ClosestPoint(pointBelow, pointsListAbove) 
                interpolatedPoint = InterpolatePoint(pointBelow, pointAbove, distanceBelow+distanceAbove, distanceBelow) 
                interpolatedPointsList.append([round(interpolatedPoint[0],2), round(interpolatedPoint[1],2), round(zValue,2)])

        for slice in contours:
            #remove the contour points for the interpolated slice
            if len(slice) > 0:
                if round(slice[0][2],2) == zValue:
                    index = contours.index(slice)
                    #add the interpolated contour points
                    contours[index] = interpolatedPointsList
                    break  

    return contours

def UnreasonableArea(contours, organ , path):
    """Determines which slices in the predicted contour have an unreasonable area.
       Unreasonable area is defined as less than the minimum area or more than the
       maximum area at a percentage through a contour. Based on the stats from the 
       PercentStats function. 

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value
        organ (str): the organ that contours were predicted for
        path (str): the path to the directory containing organogenesis folders 

    Returns:
        unreasonableArea (list): a list of z values for slices with an 
            unreasonable area

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute() 

    #get the z values in the contour
    zValueList = GetZValues(contours)

    #try to get the percent area stats if they have already been found 
    try:     
        percentAreaStats = pickle.load(open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Area Stats.txt")),'rb'))
    except: 
        Test.PercentStats(organ, path)
        percentAreaStats = pickle.load(open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Area Stats.txt")),'rb'))
  
    unreasonableArea = []

    for j, zValue in enumerate(zValueList):
        #create the list of contour points at the specified z value 
        pointsList = GetPointsAtZValue(contours, zValue)
      
        points = np.array(pointsList)
        if len(points) == 0:
            continue

        #try to create the hull. If there are too few points that are too close together, add to the unreasonable area list
        try: 
            hull = ConvexHull(points)
            #find how many percent the z value is through the contour
            percent = int(((j+1)/len(zValueList))*100)
            #tends to interpolate weird near the boundariues so exclude them
            if percent < 4 or percent > 96:
                continue
            area = abs(hull.area)
            #if the area is above the max area or below the min area, add the z value to the unreasonable area list
            for element in percentAreaStats:
                if percent == element[0]:
                    #if area < element[3]:
                    if area > element[2] or area < element[3]:
                        unreasonableArea.append(round(zValue,2))
                    break
        except:
            unreasonableArea.append(round(zValue,2))

    return unreasonableArea

def UnreasonableNumPoints(contours, organ, path):
    """Determines which slices in the predicted contour have an unreasonable number
       of points. An unreasonable number of points is defined as less than the minimum
       number of points at a specific percentage through the contour multiplied by 
       a factor. Based on the stats from the PercentStats function and factors are 
       organ specific. 

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value
        organ (str): the organ that contours were predicted for
        path (str): the path to the directory containing organogenesis folders 

    Returns:
        unreasonableNumPoints (list): a list of z values for slices with an 
            unreasonable number of points

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute() 

    zValueList = GetZValues(contours)

    #get the percent number of points stats if they haven't already been found
    try:     
        percentNumPointsStats = pickle.load(open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Area Stats.txt")),'rb'))
    except: 
        Test.PercentStats(organ, path)
        percentNumPointsStats = pickle.load(open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Points Stats.txt")),'rb'))


    #different factors work better for different organs, therefore choose the right one
    factorDictionary = {"body":0.4,"spinal cord":0.2, "oral cavity":0.2, "left parotid": 0.2, "right parotid":0.2}

    try:
        factor = factorDictionary[organ.lower()]
    except KeyError:
        #if there is no factor specified for the organ, set to default of 0.2
        factor = 0.2

    unreasonableNumPoints = []

    for j,  zValue in enumerate(zValueList): 
        pointsList = GetPointsAtZValue(contours, zValue)
        numPoints = len(pointsList)
        #find how many percent the z value is through the contour
        percent = int(((j+1)/len(zValueList))*100)
        #tends to interpolate weird near the boundariues so exclude them
        if percent < 4 or percent > 96:
            continue
        #if the number of points is less than the minimum number of points for that percentage*factor, add to the unresonable number of points list
        for element in percentNumPointsStats:
                if percent == element[0]:
                    if numPoints < element[3]*factor:
                        unreasonableNumPoints.append(round(zValue,2))
                    break

    return unreasonableNumPoints


def MissingSlices(contours, sliceThickness):
    """Determines which slices in the predicted contour are missing. Contours 
       are expected to be continuous from the lowest z value to the highest. 

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value
        sliceThickness (float): the distance between each CT slice in mm 

    Returns:
        missingZValues (list): a list of z values for the missing slices

    """ 

    zValueList = GetZValues(contours)

    startZValue = min(zValueList)
    stopZValue = max(zValueList)

    expectedZValueList = []

    i = startZValue 

    while i <= stopZValue: 
        expectedZValueList.append(i)
        i += sliceThickness 

    missingZValues = []

    #if the z value list of the predicted contour is complete then just return an empty list
    if zValueList == expectedZValueList:
        return missingZValues
    #add the missing z values to the list
    else: 
        for zValue in expectedZValueList:
            if zValue not in zValueList:
                missingZValues.append(round(zValue,2))

    return missingZValues


def GetZValues(contours):
    """Gets all of the z values in a contour. 

    Args:
        contours (list): a list of lists. Each item is a list 
            of contour points (x, y, and z values) at a specific 
            z value

    Returns:
        zValueList (list): a list of all the z values in a contour

    """ 

    zValueList = []

    for slice in contours: 
        if len(slice) > 0:
            zValueList.append(round(slice[0][2],2))
       
    return zValueList

def GetPointsAtZValue(contours, zValue): 
    """Gets all the points [x,y] at a given z value in a contour

    Args:
        contours (list): a list of lists. Each item is a list 
            of contour points (x, y, and z values) at a specific 
            z value
        zValue (float): the z value of the slice to get the 
            points for

    Returns:
        pointsList (list): a list of all the points [x,y] at 
            the given z value

    """ 

    pointsList = []

    for slice in contours:
        if len(slice) > 0:
            if round((slice[0][2]), 2) == zValue:
                for point in slice:
                    pointsList.append([point[0], point[1]])
                break

    return pointsList