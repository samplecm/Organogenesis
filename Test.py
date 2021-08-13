import numpy as np 
import matplotlib.pyplot as plt 
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
import math
from scipy.spatial import ConvexHull

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
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))) 
    model = model.to(device)    
    model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = os.listdir(dataFolder)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)
    for d in filesRange:
        imagePath = dataFiles[d]
        #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
        data = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
        data[0][0, :, :] = NormalizeImage(data[0][0, :, :])
        x = torch.from_numpy(data[0][0, :, :])
        y = torch.from_numpy(data[0][1:2, :,:])
        x = x.to(device)
        y = y.to(device)
        xLen, yLen = x.shape
        #need to reshape 
        x = torch.reshape(x, (1,1,xLen,yLen)).float()
        y = torch.reshape(y, (1,1,xLen,yLen)).float()   
        predictionRaw = (model(x)).cpu().detach().numpy()
        #now post-process the image
        x = x.cpu()
        y = y.cpu()
        x = x.numpy()
        y = y.numpy()
        prediction = PostProcessing.Process(predictionRaw[0,0,:,:], threshold)

        maskOnImage = MaskOnImage(x[0,0,:,:], prediction) #this puts the mask ontop of the CT image
        ROIOnImage = MaskOnImage(x[0,0,:,:], y[0,0,:,:])

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
        axs[1,1].imshow(prediction, cmap="gray")
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
    model.eval()

    dataPath = 'Processed_Data/' + organ #+ "_Val/" #Currently looking for patients in the test folder. 
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

def BestThreshold(organ, path, modelType, testSize=500):
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
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt")))  
    model = model.to(device)     
    model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = os.listdir(dataFolder)

    #also get the bools to find if masks are on image plane
    boolPath = 'Processed_Data/' + organ + " bools"
    boolFolder = os.path.join(path, boolPath)
    boolFiles = os.listdir(boolFolder)

    #shuffle the order (boolFiles is in same order)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)

    accuracies = [] #make a list of average accuracies calculated using different thresholds
    falsePos = []
    falseNeg = []
    fScores = []

    thresholds = np.linspace(0.05, 0.9, 11)
    
    for thres in thresholds:
        print("\nChecking Threshold: %0.3f"%(thres))
        d = 0
        #get the accuracy and F score for the current threshold value
        thresAccuracy = []
        thresFP = []
        thresFN = [] #false pos, neg
        thresFScore = []
        testSize = min(testSize, len(filesRange))
        while d < testSize:
            numStack = min(200, len(filesRange) - 1 - d)
            p = 0
            concatList = []
            while p < numStack:
                imagePath = dataFiles[d]
                image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))[0][:]
                image[0][:] = NormalizeImage(image[0][:])
                image = image.reshape((2,1,image.shape[1], image.shape[2]))
                concatList.append(image)
                p+=1
                d+=1
            if len(concatList) == 0:
                break    
            data = np.concatenate(concatList, axis=1)    #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
            print('Loaded ' + str(data.shape[1]) + ' images. Proceeding...') 
            data = torch.from_numpy(data)
            numSlices = data.shape[1]
            slices = list(range(numSlices))
            random.shuffle(slices)

            for sliceNum in slices:
                x = data[0, sliceNum, :, :]
                y = data[1, sliceNum, :, :]
                x = x.to(device)
                
                xLen, yLen = x.shape
                #need to reshape 
                x = torch.reshape(x, (1,1,xLen,yLen)).float()
                y = torch.reshape(y, (xLen,yLen)).float()   
                predictionRaw = (model(x)).cpu().detach().numpy()
                prediction = PostProcessing.Process(predictionRaw[0,0,:,:],thres) 
                prediction = np.reshape(prediction, (xLen, yLen))
                #now judge accuracy: 
               
                thresPrediction = (prediction > thres)
                imageAccuracy = np.sum(thresPrediction == y.numpy())
                imageFP = np.sum(np.logical_and(thresPrediction != y.numpy(), thresPrediction == 1))
                imageTP = np.sum(np.logical_and(thresPrediction == y.numpy(), thresPrediction == 1))
                imageTN = np.sum(np.logical_and(thresPrediction == y.numpy(), thresPrediction == 0)) 
                imageFN = np.sum(np.logical_and(thresPrediction != y.numpy(), thresPrediction == 0))    
                imageRecall = imageTP/(imageTP+imageFN)
                imagePrecision = imageTP/(imageTP+imageFP)
                if imageRecall != 0 and imagePrecision != 0 and math.isnan(imageRecall) == False and math.isnan(imagePrecision) == False:
                    imageFScore = 2.0/(imageRecall**(-1) + imagePrecision**(-1))
                    thresFScore.append(imageFScore)
                imageAccuracy *= 100/float(prediction.size) 
                imageFN *= 100 / float(prediction.size)
                imageFP *= 100 / float(prediction.size)
                thresAccuracy.append(imageAccuracy)
                thresFP.append(imageFP)
                thresFN.append(imageFN) #false pos, neg

        accuracies.append(sum(thresAccuracy) / len(thresAccuracy))
        fScores.append(sum(thresFScore)/len(thresFScore))
        falseNeg.append(sum(thresFN)/len(thresFN))
        falsePos.append(sum(thresFP)/len(thresFP))
    #Now need to determine what the best threshold to use is. Also plot the accuracy, FP, FN, F score:
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 10))

    axs[0].scatter(thresholds, accuracies)
    axs[0].plot(thresholds, accuracies)
    axs[0].set_title("Accuracy vs Threshold")
    axs[1].scatter(thresholds, falsePos)
    axs[1].plot(thresholds, falsePos)
    axs[1].set_title("False Positives vs Threshold")
    axs[2].scatter(thresholds, falseNeg)
    axs[2].plot(thresholds, falseNeg)
    axs[2].set_title("False Negatives vs Threshold")
    axs[3].scatter(thresholds, fScores)
    axs[3].plot(thresholds, fScores)
    axs[3].set_title("F Score vs Threshold")

    plt.show()    

    #Get maximimum F score
    maxFScore = max(fScores) 
    maxFScoreIndices = [i for i, j in enumerate(fScores) if j == maxFScore]
    bestThreshold = thresholds[maxFScoreIndices[0]] 
    #save this threshold
    thresFile = open(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_Thres.txt"),'w')
    thresFile.write(str(bestThreshold))
    thresFile.close()
    with open(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_Accuracy.txt"),'wb') as fp:
        pickle.dump(accuracies, fp)      
    with open(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_FalseNeg.txt"),'wb') as fp:
        pickle.dump(falseNeg, fp)   
    with open(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_FalsePos.txt"),'wb') as fp:
        pickle.dump(falsePos, fp)         
    with open(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + "_FScore.txt"),'wb') as fp:
        pickle.dump(fScores, fp)    
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
                x = contours[layer_idx][point_idx][0]
                y = contours[layer_idx][point_idx][1]
                z = contours[layer_idx][point_idx][2]
                #print([x,y,z])
                points.append([x,y,z])
                if (numPoints > 0):
                    pointList = np.vstack((pointList, np.array([x,y,z])))
                else:
                    pointList = np.array([x,y,z])   
                numPoints+=1  
              
    orig_stdout = sys.stdout            
    # with open(os.path.join(pathlib.Path(__file__).parent.absolute(),'pointCloud_LPar.txt'), 'w') as f:
    #     sys.stdout = f
    #     for point in range(len(points)):        
    #         print(str(points[point][0]) + ' ' + str(points[point][1]) + ' ' + str(points[point][2]))
    #     sys.stdout = orig_stdout    
    #Now for original contours from dicom file            
    for layer_idx in range(len(existingContours)):
        if len(existingContours[layer_idx]) > 0:
            for point_idx in range(len(existingContours[layer_idx])):
                x = existingContours[layer_idx][point_idx][0]-300
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

def FScore(organ, path, threshold, modelType):
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
    model.load_state_dict(torch.load(os.path.join(path, "Models/Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".pt"))) 
    model = model.to(device)
    model = model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = sorted(os.listdir(dataFolder))
    d = 0
    print("\nCalculating Fscore Statistics for the " + modelType + " " + organ + " model")
    TP = 0
    FP = 0
    FN = 0

    xLen,yLen = [0,0]
    while d < len(dataFiles):
        numStack = min(3, len(dataFiles) - 1 - d) 
        p = 0
        concatList = []
        while p < numStack:
            imagePath = dataFiles[d]
            image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))[0][:]
            image[0][:] = NormalizeImage(image[0][:])
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
            y = torch.from_numpy(data[1:2, sliceNum, :,:])
            x = x.to(device)
            y = y.to(device)
            xLen, yLen = x.shape
                        #need to reshape 
            x.requires_grad = True
            y.requires_grad = True
            x = torch.reshape(x, (1,1,xLen,yLen)).float()
            y = y.cpu().detach().numpy()
            y =np.reshape(y, (xLen, yLen))
 
            predictionRaw = (model(x)).cpu().detach().numpy()
            predictionRaw = np.reshape(predictionRaw, (xLen, yLen))
            prediction = PostProcessing.Process(predictionRaw, threshold)
            for x_idx in range(xLen):
                for y_idx in range(yLen):
                    if prediction[x_idx,y_idx] == 1:
                        if y[x_idx,y_idx] == 1:
                            TP += 1
                        else:
                            FP += 1
                    elif y[x_idx,y_idx] == 1:     
                        FN += 1           
        print("Finished " + str(d) + " out of " + str(1000) + " contours...")                     
    totalPoints = d * xLen * yLen                    
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F_Score = 2 / (recall**(-1) + precision ** (-1))
    accuracy = (totalPoints - FP - FN) / totalPoints
  
    return F_Score, recall, precision, accuracy

def HaussdorffDistance(organ, path, threshold, modelType):
    """Determines the 95th percentile Haussdorff distance for a model 
       with a given organ.

    Args:
        organ (str): the organ to get the Haussdorff distance for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        threshold (float): the cutoff for deciding if a pixel is 
            an organ (assigned a 1) or not (assigned a 0)
        modelType (str): the type of model

    Returns:
        float: 95th percentile Haussdorff distance

    """
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()  

    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(path, dataPath)
    dataFiles = sorted(os.listdir(dataFolder))

    patientList = []
    HaussdorffDistanceList = []
    patientCount = 0

    #want to use the patients in the validation set so get the patient names
    for file in dataFiles:
        if str(file).split("_")[1] not in patientList:
            patientList.append(str(file).split("_")[1])

    print("Calculating Haussdorff Distance for the " + modelType + " " + organ + " model...")

    for patientName in patientList: 
        predictedContourList, existingContourList, predictedContour, existingContour = Predict.GetContours(organ, patientName, path, threshold, modelType, withReal = True, tryLoad = False, plot = False)
        
        predictedContourArray = np.array(predictedContourList)
        existingContourArray = np.array(existingContourList)

        #contours are currently stored like this [x1, y1, z1, x2, y2, z2, ...] but want [(x1, y1, z1), (x2, y2, z2), ...] so reshape
        predictedContourArray = predictedContourArray.reshape(int(predictedContourArray.shape[0]/3), 3)
        existingContourArray = existingContourArray.reshape(int(existingContourArray.shape[0]/3), 3)

        #get the maximum Hausdorff distance and add to the list 
        maxHaussdorffDistance = max(directed_hausdorff(predictedContourArray, existingContourArray)[0], directed_hausdorff(existingContourArray, predictedContourArray)[0])
        HaussdorffDistanceList.append(maxHaussdorffDistance)

        patientCount += 1

        print("Finished " + str(patientCount) + " out of " + str(len(patientList)) + " patients")

    #return the 95th percentile Haussdorff distance
    return np.percentile(np.array(HaussdorffDistanceList), 95)


def GetEvalData(organ, path, threshold, modelType):
    """Creates a text file containing the hyperparameters of the model, 
       the F score data, and the 95th percentile Haussdorff distance.

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

    F_Score, recall, precision, accuracy = FScore(organ, path, threshold, modelType)
    haussdorffDistance = HaussdorffDistance(organ, path, threshold, modelType)
    haussdorffDistance = float(haussdorffDistance)

    #create a list of the hyperarameters of the model and the evaluation data
    evalData= []

    #load the hyperparamaters of the model
    with open(os.path.join(path, "Models/HyperParameters_Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".txt"), "rb") as fp:
        evalData = pickle.load(fp)

    evalData.append(["F Score", F_Score])
    evalData.append(["Recall", recall])
    evalData.append(["Precision", precision])
    evalData.append(["Accuracy", accuracy])
    evalData.append(["Threshold", threshold])
    evalData.append(["95th Percentile Haussdorff Distance", haussdorffDistance])

    #save the hyperparamters and F score data for the model 
    with open(os.path.join(path, "Evaluation Data\EvaluationData_Model_" + modelType.lower() + "_" + organ.replace(" ", "") + ".txt"), "w") as fp:
        for data in evalData:
            for element in data:
                fp.write(str(element)+ " ")
            fp.write('\n')

    return F_Score, recall, precision, accuracy, haussdorffDistance

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
    patientList = pickle.load(open(os.path.join(path, str("Processed_Data/Sorted Contour Lists/" + organ + " Good Contours.txt")), 'rb')) 

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
                if (zValue == contourArray[i,2]):
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
