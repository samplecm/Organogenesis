import numpy as np 
import matplotlib.pyplot as plt
from numpy.lib.function_base import average, percentile 
import Model
import Predict
import DicomParsing
import torch
import torch.nn as nn
import pickle
import random
import os
import pathlib
import PostProcessing
import cv2 as cv
import open3d as o3d
import plotly.graph_objects as graph_objects
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from scipy.spatial.distance import directed_hausdorff
from DicomParsing import GetDICOMContours
import math
from scipy.spatial import ConvexHull

def GetTrainingVolumeStats(roi, path):
    """Gets the average volume of all organ ROIs in the patient files.
    
    Args:
        organ (str): the name of the roi which is of interest
        path (str): path to the directory containing patient directories.
        return: None

        Produces a text file in the program directory's "Statistics" folder

    """
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   
    sliceSeparations = [] #Keep track of slice separation and send error message if there is more than one (gaps .. islands?)    
    num_rois = 0
    volumes = []
    patientFiles = os.listdir(os.path.join(path, "Patient_Files"))

    for file in patientFiles:
        

        patientVolume = 0
        contours = GetDICOMContours(file, roi, path)
        if len(contours) == 0:
            continue
        print("Loading patient: " + file + " for volume calculation.")    
        for slice_idx in range(len(contours)-1):
            #create the list of contour points at the specified z value 
            slice = contours[slice_idx]
            if len(slice) == 0:
                continue
            sliceAbove = contours[slice_idx+1]
            if len(sliceAbove) == 0:
                continue
            sliceArea = Area_of_Slice(slice)
            sliceAboveArea = Area_of_Slice(sliceAbove)
            sliceSeparation = abs(contours[slice_idx+1][0][2] - contours[slice_idx][0][2])
            patientVolume = patientVolume + (sliceSeparation *(sliceArea + sliceAboveArea) / 2)

            if sliceSeparation not in sliceSeparations:
                sliceSeparations.append(sliceSeparation)
            volumes.append(patientVolume)    

    if len(sliceSeparations) > 1:
        print("Multiple separations in adjacent contour slices found. Islands may be affecting the volume calculation for patient: " + file)
    if volumes == []:
        raise Exception("No organ volumes were calculated.")           
    averageVolume = 0
    numVolumes = 0      
    for volume in volumes:
        if volume != 0:
            numVolumes = numVolumes + 1
            averageVolume = averageVolume + volume
    averageVolume = averageVolume / numVolumes
    with open(os.path.join(path, "Statistics", str(roi.replace(" ", "") + "_volume.txt")), "w") as fp:
        fp.write(str(averageVolume))
    return averageVolume        



def Area_of_Slice(slice):
    if len(slice) == 0:
        return 0
    pointsList = []
    for i in range(len(slice)):
        pointsList.append([slice[i][0], slice[i][1]])
    points = np.array(pointsList)

    hull = ConvexHull(points)
    area = hull.area

    return area

def PlotTrainingMasks(organList, path): 
    #currently works for tubarial glands
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = os.getcwd()  
    
    dataPaths = []
    fileLists = []
    for organ in organList: 
        dataPaths.append(os.path.join(path, str('Processed_Data/' + organ + "_Val/")))
        fileLists.append(os.listdir(dataPaths[-1]))
    
    for file in fileLists[0]:    
        maskImage = None    
        masks = None
        for dataPath in dataPaths:
            data = pickle.load(open(os.path.join(dataPath, file), 'rb'))
            if np.amax(data[0][1,:,:]) == 0:
                break
            x = data[0][0, :, :]
            y = data[0][1, :,:]    
            if maskImage is None:       
                maskImage = MaskOnImage(x[:,:], y[:,:])
                masks = y

            else: 
                maskImage = MaskOnImage(maskImage, y[:,:])    
                masks = MaskOnImage(masks, y[:,:])  
            
        if np.amax(data[0][1,:,:]) == 0:
            continue
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        #predictedContour = PostProcessing.MaskToContour(prediction[0,0,:,:])
        #contour = PostProcessing.MaskToContourImage(y[0,0,:,:])
        axs[0,0].imshow(x[:,:], cmap='gray')
        axs[0,0].set_title("Original Image")
        #axs[0,1].hist(predictionRaw[0,0,:,:], bins=5)
        axs[0,1].imshow(maskImage[:,:] , cmap='gray')
        axs[0,1].set_title("Mask on Image")
        axs[1,0].imshow(masks[:,:], cmap='gray')
        axs[1,0].set_title("Original Mask")
        print(file)

        
        # axs[2,0].imshow(y[0,1, :,:], cmap = "gray")
        # axs[2,0].set_title("Original Background")
        # axs[2,1].imshow(prediction[0,1, :,:], cmap = "gray")
        # axs[2,1].set_title("Predicted Background")
        plt.show()




def TestPlot(organ, path, threshold, modelType):
    """Plots 2D CTs with both manually drawn and predicted masks 
       for visual comparison.

    Args:
        organ (str): the organ to plot masks for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float) : the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models", organ, "Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))) 
    model = model.to(device)    
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = os.listdir(dataFolder)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)
    for d in filesRange:
        imagePath = dataFiles[d]
        #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
        data = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
        if np.amax(data[0][1,:,:]) == 0:
            continue
        #print(np.amax(data[0][1,:,:]))
        x = torch.from_numpy(data[0][0, :, :])
        y = torch.from_numpy(data[0][1, :,:])
        
        x = x.to(device)
        y = y.to(device)
        xLen, yLen = x.shape
        #need to reshape 
        x = torch.reshape(x, (1,1,xLen,yLen)).float()
        y = torch.reshape(y, (1,1,xLen,yLen)).float()   
        prediction = (model(x)).cpu().detach().numpy()
        #now post-process the image
        x = x.cpu()
        y = y.cpu()
        x = x.numpy()
        y = y.numpy()
        # prediction = prediction[0,0,:,:]
        # print(np.amax(prediction))
        prediction = PostProcessing.Process(prediction[0,0,:,:], threshold, modelType, organ)
        

        maskOnImage = MaskOnImage(x[0,0,:,:], prediction) #this puts the mask ontop of the CT image
        ROIOnImage = MaskOnImage(x[0,0,:,:], y[0,0,:,:])

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
        
        #predictedContour = PostProcessing.MaskToContour(prediction[0,0,:,:])
        #contour = PostProcessing.MaskToContourImage(y[0,0,:,:])
        axs[0,0].imshow(ROIOnImage)
        axs[0,0].set_title("Original Image")
        #axs[0,1].hist(predictionRaw[0,0,:,:], bins=5)
        axs[0,1].imshow(maskOnImage)
        axs[0,1].set_title("Mask on Image")
        axs[1,0].imshow(y[0,0, :,:])
        axs[1,0].set_title("Original Mask")
        axs[1,1].imshow(prediction)
        axs[1,1].set_title("Predicted Mask")

        
        # axs[2,0].imshow(y[0,1, :,:], cmap = "gray")
        # axs[2,0].set_title("Original Background")
        # axs[2,1].imshow(prediction[0,1, :,:], cmap = "gray")
        # axs[2,1].set_title("Predicted Background")
        plt.show()
        
def GetMasks(organ, patientName, path, threshold, modelType):
    """Gets the existing and predicted masks of a given organ for a given patient. 

    Args:
        organ (str): the organ to predict contours for
        patientName (str): the name of the patient folder conating dicom files 
            (CT images) to get masks for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float) : the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    Returns:
        predictionsArray (3D numpy array): predicted masks of the organ for 
            a given patient
        exisitingArray (3D numpy array): existing masks of the organ for
            a given patient

    """

    #Returns a 3d array of predicted masks for a given patient name. 
    if path == None:
        path = pathlib.Path(__file__).parent.absolute()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt")))    
    model = model.to(device)    

    dataPath = 'Processed_Data/' + organ + "_Test/" #Currently looking for patients in the test folder. 
    dataFolder = os.path.join(path, dataPath)
    dataFiles = os.listdir(dataFolder)
    filesRange = list(range(len(dataFiles)))

    patientImages = []

    for d in filesRange: #First get the files for the patientName given
        if patientName in dataFiles[d]:
            patientImages.append(dataFiles[d])
    patientImages.sort()    

    predictions = [] #First put 2d masks into predictions list and then at the end stack into a 3d array
    existingMasks = []
    for image in patientImages:
        #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
        data = pickle.load(open(os.path.join(dataFolder, image), 'rb'))[0][:]
        data[0][:] = NormalizeImage(data[0][:])
        x = torch.from_numpy(data[0][:])
        y = torch.from_numpy(data[1][:])
        x = x.to(device)
        xLen, yLen = x.shape
        #need to reshape 
        x = torch.reshape(x, (1,1,xLen,yLen)).float()
        predictionRaw = (model(x)).cpu().detach().numpy()
        #now post-process the image
        x = x.cpu()
        x = x.numpy()
        predic = PostProcessing.Process(predictionRaw[0,0,:,:], threshold)
        predictions.append(predic)
        existingMasks.append(y)
    #Stack into 3d array    
    predictionsArray = np.stack(predictions, axis=0)
    existingArray = np.stack(existingMasks, axis=0)
    return predictionsArray, existingArray    

def MaskOnImage(image, mask):
    """takes a normalized CT image and a binary mask, and creates a normalized 
    image with the mask on the CT.

    Args:
        image (2D numpy array): the CT image which the mask image corresponds to.
        mask (2D numpy array): the binary mask which is to be placed on top of the CT image.

    Returns:
        (2D numpy array): a normalized image of the mask placed on top of the CT image, 
            with the binary mask pixel value being 1.2x the maximum CT pixel value.

    """
    xLen, yLen = image.shape
    image = NormalizeImage(image)
    for x in range(xLen):
        for y in range(yLen):
            if mask[x,y] == 1:
                image[x,y] = 1.2
    return NormalizeImage(image)            

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

def BestThreshold(organ, path, modelType, test_size=1000):
    """Determines the best threshold for predicting contours based on the 
       F score. Displays graphs of the accuracy, false positives, false
       negatives, and F score vs threshold. Also saves these stats to the
       model folder.

    Args:
        organ (str): the organ to get the best threshold for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        modelType (str): the type of model
        testSize (str, int): the number of images to test with for each threshold

    Returns: 
        bestThreshold (float): the best threshold for predicting contours based on 
            F score

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice being used: " + device.type)
    print("\nDetermining most accurate threshold for the " + modelType + " " + organ + " model...")
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models/", organ,"Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt")))  
    model = model.to(device)     
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = os.listdir(dataFolder)


    #shuffle the order (boolFiles is in same order)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)

    accuracies = [] #make a list of average accuracies calculated using different thresholds
    recalls = []
    precisions = []
    f_scores = []

    thresholds = np.linspace(0.3, 0.7, 15)
    
    for thres in thresholds:
        print("\nChecking Threshold: %0.3f"%(thres))
        files_max = test_size
        F_Score, recall, precision, accuracy = FScore(organ, path,thres, modelType,test_size=files_max)
        f_scores.append(F_Score)
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)
        
    #Now need to determine what the best threshold to use is. Also plot the accuracy, FP, FN, F score:
    # fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

    # axs[0].scatter(thresholds, accuracies)
    # axs[0].plot(thresholds, accuracies)
    # axs[0].set_title("Accuracy vs Threshold")
    # axs[1].scatter(thresholds, recalls)
    # axs[1].plot(thresholds, recalls)
    # axs[1].set_title("Recall vs Threshold")
    # axs[2].scatter(thresholds, precisions)
    # axs[2].plot(thresholds, precisions)
    # axs[2].set_title("Precision vs Threshold")
    # axs[3].scatter(thresholds, f_scores)
    # axs[3].plot(thresholds, f_scores)
    # axs[3].set_title("F Score vs Threshold")

    # plt.show()    

    #Get maximimum F score
    best_index = f_scores.index(max(f_scores) )
    bestThreshold = thresholds[best_index] 
    print(f"Best threshold found: {thresholds[best_index]}")
    #save this threshold
    thresFile = open(os.path.join(path, "Models", organ, "Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_Thres.txt"),'w')
    thresFile.write(str(bestThreshold))
    thresFile.close()
   
    print(f"best threshold: {bestThreshold}")    
    return bestThreshold



    # numPoints = 0
    # #turn contours into separate x,y,z lists of lists
    # X = []
    # Y = []
    # Z = []
    # for layer_idx in range(len(contours)):
    #     if len(contours[layer_idx]) > 0:
    #         X.append([])
    #         Y.append([])
    #         Z.append([])
    #         for point_idx in range(len(contours[layer_idx])):
    #             x = contours[layer_idx][point_idx][0]
    #             y = contours[layer_idx][point_idx][1]
    #             z = layer_idx * 2.5
    #         X[-1].append(x)
    #         Y[-1].append(y)
    #         Z[-1].append(z)  
   

    # print(numPoints)              


    # fig = graph_objects.Figure(data=[graph_objects.Mesh3d(x=pointList[:,0], y = pointList[:,1], z = pointList[:,2], alphahull=5, opacity=0.5, color='cyan')])
    # fig.update_xaxes(title_text="x")
    # fig.update_yaxes(title_text="y")
    # fig.update_xaxes(range=[0,512])
    # fig.update_yaxes(range=[0,512])

    # fig.show()
    # fig = plt.figure() 
    # ax = fig.add_subplot(111, projection='3d')
    # cset = ax.contour(X,Y,Z, cmap=cm.coolwarm)
    # ax.clabel(cset, fontsize=9, inline=1)
    # plt.show() 
    # pointCloud.colors = o3d.utility.Vector3dVector(pointList[:,3:6]/255)
    # pointCloud.normals = o3d.utility.Vector3dVector(pointList[:,6:9])
    
    #Now need to make a numpy array as a list of points

def PlotPatientContours(contours, existingContours):
    """Plots the contours provided.   

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value
        existingContours (list): a list of lists. Each item is a
            list of the existing contour points at a specific 
            z value.

    """
    pointCloud = o3d.geometry.PointCloud()
    numPoints = 0
    points = []
    import sys
    for layer_idx in range(len(contours)):
        if len(contours[layer_idx]) > 0:
            for point_idx in range(len(contours[layer_idx])):
                try:
                    x = contours[layer_idx][point_idx][0]
                    y = contours[layer_idx][point_idx][1]
                    z = contours[layer_idx][point_idx][2]
                except: continue    
                #print([x,y,z])
                points.append([x,y,z])
                if (numPoints > 0):
                    pointList = np.vstack((pointList, np.array([x,y,z])))
                else:
                    pointList = np.array([x,y,z])   
                numPoints+=1  
              
    orig_stdout = sys.stdout            
    if len(existingContours) == 0:
        print("Existing contours not found. Plotting only predicted contours.")  
        pointCloud.points = o3d.utility.Vector3dVector(pointList[:,:3])
        o3d.visualization.draw_geometries([pointCloud])
        return
    #Now for original contours from dicom file            
    for layer_idx in range(len(existingContours)):
        if len(existingContours[layer_idx]) > 0:
            for point_idx in range(len(existingContours[layer_idx])):
                x = existingContours[layer_idx][point_idx][0] - 20
                y = existingContours[layer_idx][point_idx][1]
                z = existingContours[layer_idx][point_idx][2]
                
                #print([x,y,z])
                if (numPoints > 0):
                    pointList = np.vstack((pointList, np.array([x,y,z])))
                else:
                    pointList = np.array([x,y,z])   
                numPoints+=1       

    pointCloud.points = o3d.utility.Vector3dVector(pointList[:,:3])
    o3d.visualization.draw_geometries([pointCloud])

def FScore(organ, path, threshold, modelType, test_size=None):
    """Computes the F score, accuracy, precision, and recall for a model
       on a given organ. Uses the validation data. 

    Args:
        organ (str): the organ to get the F score for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float) : the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    Returns:
        f_Score (float): (2 * precision * recall) / (precision + recall)
        recall (float): truePositives / (truePositives + falseNegatives)
        precision (float): truePositives / (truePositives + falsePositives)
        accuracy (float): (truePositives + trueNegatives) / allPixels
    """
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice being used for computing F Score: " + device.type)
    if modelType.lower() == "unet":
        model = Model.UNet()
    elif modelType.lower() == "multiresunet": 
        model = Model.MultiResUNet()
    model.load_state_dict(torch.load(os.path.join(path, "Models", organ, "Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))) 
    model = model.to(device)
    #model = model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    data_files = sorted(os.listdir(dataFolder))
    d = 0
    print("\nCalculating Fscore Statistics for the " + modelType + " " + organ + " model")
    TP = 0
    FP = 0
    FN = 0

    xLen,yLen = [0,0]
    if type(test_size) == int:
        num_files = min(test_size, len(data_files))
    else:
        num_files = len(data_files)    
    while d < num_files:
        numStack = min(10, num_files - 1 - d) 
        p = 0
        concatList = []
        while p < numStack:
            imagePath = data_files[d]
            image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))[0][:]
            image = image.reshape((2,1,image.shape[2], image.shape[2]))
            concatList.append(image)
            p+=1
            d+=1
        if len(concatList) == 0:
            break    
        data = np.concatenate(concatList, axis=1)  
        numSlices = data.shape[1]
        for sliceNum in range(numSlices):
            x = torch.from_numpy(data[0, sliceNum, :, :])
            y = torch.from_numpy(data[1, sliceNum, :,:])
            x = x.to(device)
            y = y.to(device)
            xLen, yLen = x.shape
                        #need to reshape 
            x = torch.reshape(x, (1,1,xLen,yLen)).float()
            y = y.cpu().detach().numpy()
            y =np.reshape(y, (xLen, yLen))
 
            predictionRaw = (model(x)).cpu().detach().numpy()
            predictionRaw = np.reshape(predictionRaw, (xLen, yLen))
            prediction = PostProcessing.Process(predictionRaw, threshold, modelType, organ)
            for x_idx in range(xLen):
                for y_idx in range(yLen):
                    if prediction[x_idx,y_idx] == 1:
                        if y[x_idx,y_idx] == 1:
                            TP += 1
                        else:
                            FP += 1
                    elif y[x_idx,y_idx] == 1:     
                        FN += 1           
        print("Finished " + str(d) + " out of " + str(num_files) + " contours...")                     
    totalPoints = d * xLen * yLen                    
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F_Score = 2 / (recall**(-1) + precision ** (-1))
    accuracy = (totalPoints - FP - FN) / totalPoints
  
    return F_Score, recall, precision, accuracy

def HausdorffDistance(organ, path, threshold, modelType):
    """Determines the 95th percentile Hausdorff distance for a model 
       with a given organ.

    Args:
        organ (str): the organ to get the Hausdorff distance for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    Returns:
        float: 95th percentile Hausdorff distance

    """
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()  

    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = sorted(os.listdir(dataFolder))

    patientList = []
    HausdorffDistanceList = []
    patientCount = 0

    #want to use the patients in the validation set so get the patient names
    for file in dataFiles:
        splitFile = file.split("_")
        fileName = splitFile[1]
        for s in range(2, len(splitFile)-1):
            fileName = fileName + "_" + splitFile[s]
        if fileName not in patientList:          
            patientList.append(fileName)

    print("Calculating Hausdorff Distance for the " + modelType + " " + organ + " model...")
    if len(patientList) > 20:
        patientList = patientList[0:21]
    for patientName in patientList: 
        predictedContour, existingContour, predictedContourList, existingContourList = Predict.GetContours(organ, patientName, path, threshold, modelType, withReal = True, tryLoad = False, plot=False)
        
        predictedContourArray = np.array(predictedContourList)
        existingContourArray = np.array(existingContourList)

        #contours are currently stored like this [x1, y1, z1, x2, y2, z2, ...] but want [(x1, y1, z1), (x2, y2, z2), ...] so reshape
        predictedContourArray = predictedContourArray.reshape(int(predictedContourArray.shape[0]/3), 3)
        existingContourArray = existingContourArray.reshape(int(existingContourArray.shape[0]/3), 3)

        #get the maximum Hausdorff distance and add to the list 
        for i in range(predictedContourArray.shape[0]): 
            haus1 = directed_hausdorff(np.array([predictedContourArray[i,:]]), existingContourArray)[0]
            #haus2 = directed_hausdorff(existingContourArray, np.array([predictedContourArray[i,:]]))[0]
            HausdorffDistanceList.append(haus1)
        for i in range(existingContourArray.shape[0]): 
            haus1 = directed_hausdorff(np.array([existingContourArray[i,:]]), predictedContourArray)[0]
            #haus2 = directed_hausdorff(existingContourArray, np.array([predictedContourArray[i,:]]))[0]
            HausdorffDistanceList.append(haus1)    


        patientCount += 1

        print("Finished " + str(patientCount) + " out of " + str(len(patientList)) + " patients")

    #return the 95th percentile Hausdorff distance
    percentile_haus = np.percentile(np.array(HausdorffDistanceList), 95)
    return percentile_haus


def GetEvalData(organ, path, threshold, modelType):
    """Creates a text file containing the hyperparameters of the model, 
       the F score data, and the 95th percentile Hausdorff distance.

    Args:
        organ (str): the organ to get evaluation data for 
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()  

    hausdorffDistance = HausdorffDistance(organ, path, threshold, modelType)
    hausdorffDistance = float(hausdorffDistance)

    F_Score, recall, precision, accuracy = FScore(organ, path, threshold, modelType)
    print("F Score: " + str(F_Score))
    print("Hausdorff: " + str(hausdorffDistance))
    

    #create a list of the hyperarameters of the model and the evaluation data
    evalData= []

    #load the hyperparamaters of the model
    try:
        with open(os.path.join(path, "Models/HyperParameters_Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".txt"), "rb") as fp:
            evalData = pickle.load(fp)
    except:
        pass        
        


    evalData.append(["F Score", F_Score])
    evalData.append(["Recall", recall])
    evalData.append(["Precision", precision])
    evalData.append(["Accuracy", accuracy])
    evalData.append(["Threshold", threshold])
    evalData.append(["95th Percentile Hausdorff Distance", hausdorffDistance])

    #save the hyperparamters and F score data for the model 
    with open(os.path.join(path, "Evaluation Data\EvaluationData_Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".txt"), "w") as fp:
        for data in evalData:
            for element in data:
                fp.write(str(element)+ " ")
            fp.write('\n')
    
    return F_Score, recall, precision, accuracy, hausdorffDistance

def PercentStats(organ, path):
    """Saves lists of the area and the number of contour points at each percentage 
       through a contour. Saves the max, min, and average area and number of contour 
       points at each percentage through the contour. Gets the data from patient 
       data sorted into the good contours list.

    Args:
        organ (str): the organ to get evaluation data for 
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()  

    #get the patient list from the sorted contour lists
    try:
        patientList = pickle.load(open(os.path.join(path, str("Processed_Data/Sorted Contour Lists/" + organ + " Good Contours.txt")), 'rb')) 
    except: 
        patientList = os.listdir(os.path.join(path, "Patient_Files"))
    percentAreaList = []
    percentNumPointsList = []

    for patientCount, patientName in enumerate(patientList): 
        #get the existing contours for each patient
        contourData = Predict.GetOriginalContours(organ, patientName, path)
        contour = contourData[0]
        contourList = contourData[1]
        contourArray = np.array(contourList)

        #contours are currently stored like this [x1, y1, z1, x2, y2, z2, ...] but want [(x1, y1, z1), (x2, y2, z2), ...] so reshape
        contourArray = contourArray.reshape(int(contourArray.shape[0]/3), 3)

        #get the z values in the contour
        zValueList = PostProcessing.GetZValues(contour)

        for j, zValue in enumerate(zValueList):
            #create the list of contour points at the specified z value 
            pointsList = []
            for i in range(contourArray.shape[0]):
                if (round(zValue,2) - round(contourArray[i,2],2) <= 0.1):
                    pointsList.append([contourArray[i,0], contourArray[i,1]])
            points = np.array(pointsList)
            numPoints = len(pointsList)

            #create the hull
            hull = ConvexHull(points)

            #find how many percent the z value is through the contour
            percent = int(((j+1)/len(zValueList))*100)

            #append the percent through the contour and the area of the contour to the list 
            percentAreaList.append([percent, hull.area])
            percentNumPointsList.append([percent, numPoints])

        print("Finished " + str(patientCount + 1) + " out of " + str(len(patientList)) + " patients")

    with open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Area List.txt")), "wb") as fp:
        pickle.dump(percentAreaList, fp)  
     
    with open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Points List.txt")), "wb") as fp:
        pickle.dump(percentNumPointsList, fp)  

    percentAreaStats = []
    percentNumPointsStats = []

    for percentIndex in range(101): 
        areaList = []
        numPointsList = []
        for percentArea in percentAreaList:
            if percentArea[0] == percentIndex:
                areaList.append(percentArea[1])
        if len(areaList) > 0:
            averageArea = sum(areaList)/(len(areaList))
            maxArea = max(areaList)
            minArea = min(areaList)
            percentAreaStats.append([percentIndex, averageArea, maxArea, minArea])

        for percentNumPoints in percentNumPointsList:
            if percentNumPoints[0] == percentIndex: 
                numPointsList.append(percentNumPoints[1])
        if len(numPointsList) > 0:
            averageNumPoints = sum(numPointsList)/len(numPointsList)
            minNumPoints = min(numPointsList)
            maxNumPoints = max(numPointsList)
            percentNumPointsStats.append([percentIndex, averageNumPoints, maxNumPoints, minNumPoints])

    with open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Area Stats.txt")), "wb") as fp:
        pickle.dump(percentAreaStats, fp) 

    with open(os.path.join(path, str("Processed_Data/Area Stats/" + organ + " Percent Points Stats.txt")), "wb") as fp:
        pickle.dump(percentNumPointsStats, fp) 
