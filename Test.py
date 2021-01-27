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
import open3d as o3d
import plotly.graph_objects as graph_objects
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm


def TestPlot(organ, threshold):
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))    
    model = model.cuda()      
    model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath)
    dataFiles = os.listdir(dataFolder)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)
    for d in filesRange:
        imagePath = dataFiles[d]
        #data has 4 dimensions, first is the type (image, contour, background), then slice, and then the pixels.
        data = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
        print("Validating with " + dataFiles[d])
        x = torch.from_numpy(data[0, :, :]).cuda()
        y = torch.from_numpy(data[1:2, :,:]).cuda()
        xLen, yLen = x.shape
        #need to reshape 
        x = torch.reshape(x, (1,1,xLen,yLen)).float()
        y = torch.reshape(y, (1,1,xLen,yLen)).float()   
        predictionRaw = (model(x)).cpu().detach().numpy()
        prediction = PostProcessing.Process(predictionRaw, threshold)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        maskOnImage = MaskOnImage(x[0,0,:,:], prediction[0,0,:,:])
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
        axs[1,1].imshow(prediction[0,0, :,:], cmap="gray")
        axs[1,1].set_title("Predicted Mask")

        
        # axs[2,0].imshow(y[0,1, :,:], cmap = "gray")
        # axs[2,0].set_title("Original Background")
        # axs[2,1].imshow(prediction[0,1, :,:], cmap = "gray")
        # axs[2,1].set_title("Predicted Background")
        plt.show()
        


def MaskOnImage(image, mask):
    xLen, yLen = image.shape
    image = NormalizeImage(image)
    for x in range(xLen):
        for y in range(yLen):
            if mask[x,y] == 1:
                image[x,y] = 1.2
    return NormalizeImage(image)            


def NormalizeImage(image):
    #if its entirely ones or entirely zeros, can just return it as is. 
    if np.amin(image) == 1 and np.amax(image) == 1:
        return image
    elif np.amin(image) == 0 and np.amax(image) == 0:
        return image     
    ptp = np.ptp(image)
    amin = np.amin(image)
    return (image - amin) / ptp    

def Best_Threshold(organ, testSize=10e6, onlyMasks=False, onlyBackground=False):
    #return the model output threshold that maximizes accuracy. onlyMasks if true will calculate statistics
    #only for images that have a mask on the plane, and onlyBackground will do it for images without masks.
    print("Determining most accurate threshold...")
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))    
    model = model.cuda()      
    model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath)
    dataFiles = os.listdir(dataFolder)

    #also get the bools to find if masks are on image plane
    boolPath = 'Processed_Data/' + organ + " bools"
    boolFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), boolPath)
    boolFiles = os.listdir(boolFolder)

    #shuffle the order (boolFiles is in same order)
    filesRange = list(range(len(dataFiles)))
    random.shuffle(filesRange)

    
    accuracies = [] #make a list of average accuracies calculated using different thresholds
    falsePos = []
    falseNeg = []
    thresholds = np.linspace(0.05,0.7,15)
    
    for thres in thresholds:
        print("Checking Threshold: %0.3f"%(thres))
        d = 0
        #get the accuracy for the current threshold value
        thresAccuracy = []
        thresFP = []
        thresFN = [] #false pos, neg
        testSize = min(testSize, len(filesRange))
        while d < testSize:
            numStack = min(200, len(filesRange) - 1 - d)
            p = 0
            concatList = []
            while p < numStack:
                imagePath = dataFiles[d]
                if onlyMasks:
                    boolMask = pickle.load(open(os.path.join(boolFolder, boolFiles[d]), 'rb'))
                    if not boolMask:
                        p+=1
                        d+=1
                        continue
                elif onlyBackground:
                    boolMask = pickle.load(open(os.path.join(boolFolder, boolFiles[d]), 'rb'))
                    if boolMask:
                        p+=1
                        d+=1
                        continue    
                image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
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
                x = data[0, sliceNum, :, :].cuda()
                y = data[1, sliceNum, :, :]
                xLen, yLen = x.shape
                #need to reshape 
                x = torch.reshape(x, (1,1,xLen,yLen)).float()
                y = torch.reshape(y, (xLen,yLen)).float()   
                predictionRaw = (model(x)).cpu().detach().numpy()
                prediction = PostProcessing.Process(predictionRaw,thres) 
                prediction = np.reshape(prediction, (xLen, yLen))
                #now judge accuracy: 
               
                thresPrediction = (prediction > thres)
                imageAccuracy = np.sum(thresPrediction == y.numpy())
                imageFP = np.sum(np.logical_and(thresPrediction != y.numpy(), thresPrediction == 1))
                imageFN = np.sum(np.logical_and(thresPrediction != y.numpy(), thresPrediction == 0))         
                imageAccuracy *= 100/float(prediction.size) 
                imageFN *= 100 / float(prediction.size)
                imageFP *= 100 / float(prediction.size)
                thresAccuracy.append(imageAccuracy)
                thresFP.append(imageFP)
                thresFN.append(imageFN) #false pos, neg

        accuracies.append(sum(thresAccuracy) / len(thresAccuracy))
        falseNeg.append(sum(thresFN)/len(thresFN))
        falsePos.append(sum(thresFP)/len(thresFP))
    #Now need to determine what the best threshold to use is. Also plot the accuracy, FP, FN:
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))

    axs[0].scatter(thresholds, accuracies)
    axs[0].plot(thresholds, accuracies)
    axs[0].set_title("Accuracy vs Threshold")
    axs[1].scatter(thresholds, falsePos)
    axs[1].plot(thresholds, falsePos)
    axs[1].set_title("False Positives vs Threshold")
    axs[2].scatter(thresholds, falseNeg)
    axs[2].plot(thresholds, falseNeg)
    axs[2].set_title("False Negatives vs Threshold")

    plt.show()    

    #Get maximimum accuracy 
    maxAccuracy = max(accuracies) 
    maxAccuracyIndices = [i for i, j in enumerate(accuracies) if j == maxAccuracy]
    bestThreshold = thresholds[maxAccuracyIndices[0]] 
    #save this threshold
    thresFile = open(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + "_Thres.txt"),'w')
    thresFile.write(str(bestThreshold))
    thresFile.close()
    with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + "_Accuracy.txt"),'wb') as fp:
        pickle.dump(accuracies, fp)      
    with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + "_FalseNeg.txt"),'wb') as fp:
        pickle.dump(falseNeg, fp)   
    with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + "_FalsePos.txt"),'wb') as fp:
        pickle.dump(falsePos, fp)            
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
    #overload having existing patient contour to plot as well
    pointCloud = o3d.geometry.PointCloud()
    numPoints = 0
    points = []
    import sys
    for layer_idx in range(len(contours)):
        if len(contours[layer_idx]) > 0:
            for point_idx in range(len(contours[layer_idx])):
                x = contours[layer_idx][point_idx][0]
                y = contours[layer_idx][point_idx][1]
                z = layer_idx * 2.5
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
                x = existingContours[layer_idx][point_idx][0] - 30
                y = existingContours[layer_idx][point_idx][1] - 30
                z = layer_idx * 2.5
                
                if (numPoints > 0):
                    pointList = np.vstack((pointList, np.array([x,y,z])))
                else:
                    pointList = np.array([x,y,z])   
                numPoints+=1              
            # print(z)    
    print(pointList[0,:])
    print(numPoints)              
    pointCloud.points = o3d.utility.Vector3dVector(pointList[:,:3])
    o3d.visualization.draw_geometries([pointCloud])

def FScore(organ, threshold):
    model = Model.UNet()
    model.load_state_dict(torch.load(os.path.join(pathlib.Path(__file__).parent.absolute(), "Models/Model_" + organ.replace(" ", "") + ".pt")))  
    model = model.cuda()
    model = model.eval()
    dataPath = 'Processed_Data/' + organ + "_Val/"
    dataFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), dataPath)
    dataFiles = sorted(os.listdir(dataFolder))
    d = 0
    print('Calculating Statistics')
    TP = 0
    FP = 0
    FN = 0

    xLen,yLen = [0,0]
    while d <  150:#len(dataFiles):
        numStack = min(3, len(dataFiles) - 1 - d) #loading 1400 images at a time (takes about 20GB RAM)
        p = 0
        concatList = []
        while p < numStack:
            imagePath = dataFiles[d]
            image = pickle.load(open(os.path.join(dataFolder, imagePath), 'rb'))
            image = image.reshape((2,1,image.shape[2], image.shape[2]))
            concatList.append(image)
            p+=1
            d+=1
        if len(concatList) == 0:
            break    
        data = np.concatenate(concatList, axis=1)  
        numSlices = data.shape[1]
        for sliceNum in range(numSlices):
            x = torch.from_numpy(data[0, sliceNum, :, :]).cuda()
            y = torch.from_numpy(data[1:2, sliceNum, :,:]).cuda()
            xLen, yLen = x.shape
            #need to reshape 
            x.requires_grad = True
            y.requires_grad = True
            x = torch.reshape(x, (1,1,xLen,yLen)).float()
            y = torch.reshape(y, (1,1,xLen,yLen)).float()
            predictionRaw = (model(x)).cpu().detach().numpy()
            prediction = PostProcessing.Process(predictionRaw, threshold)
            for x_idx in range(xLen):
                for y_idx in range(yLen):
                    if prediction[0,0,x_idx,y_idx] == 1:
                        if y[0,0,x_idx,y_idx] == 1:
                            TP += 1
                        else:
                            FP += 1
                    elif y[0,0,x_idx,y_idx] == 1:     
                        FN += 1  
            print("Finished a patient")            
        print("Finished " + str(d) + "patients...")                     
    totalPoints = d * xLen * yLen                    
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F_Score = 2 / (recall**(-1) + precision ** (-1))
    accuracy = (totalPoints - FP - FN) / totalPoints
    return F_Score, recall, precision, accuracy

        