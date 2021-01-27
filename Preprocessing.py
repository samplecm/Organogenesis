import numpy as np 
import torch
import torch.nn as nn
import cv2 as cv
import matplotlib.pyplot as plt
import math
import cv2 as cv
import open3d as o3d
import plotly.graph_objects as graph_objects
from mpl_toolkits.mplot3d import Axes3D 

def Process(combinedImages):
#Take patient CTs and contours and then remove any uncontoured areas images that are in the middle of an ROI
#This assumes that the structure is fully connected
    firstImageFound = False
    idx = 0
    while idx < combinedImages.shape[1]:
        if firstImageFound and np.amax(combinedImages[1,idx,:,:]) == 0:
            #Now see if there are still contours above this. If so, delete this slice from the array.
            above_idx = idx + 1
            while above_idx < combinedImages.shape[1]:
                if np.amax(combinedImages[1,idx,:,:]) > 0:
                    #remove the slice
                    combinedImages = np.delete(combinedImages, idx, 1)
                    print("Removed uncontoured image")
                    break
                above_idx +=1

        #if this is the first contour, mark that the first has been found
        elif np.amax(combinedImages[1,idx,:,:]) > 0:
            firstImageFound = True

        idx+=1    
    
    return combinedImages

def PlotContours(points):
    print("Plotting")
    #Here we plot a 3d image of the pointcloud from the list of masks. 
    #First need to convert the masks to contours: 
    pointCloud = o3d.geometry.PointCloud()
    pointCloud.points = o3d.utility.Vector3dVector(pointList[:,:3])
    o3d.visualization.draw_geometries([pointCloud])
         




