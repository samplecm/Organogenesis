import numpy as np 
import pydicom
import pickle
import os
from rtstruct_builder import RTStructBuilder
from rtstruct import RTStruct
from rtutils import ROIData
import ds_helper
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
import pathlib
import random

def SaveToDICOM(patientName, organList, path, contoursList):
    """Uses a pre trained model to predict contours for a given organ. Saves 
       the contours to the Predictions_Patients folder in a binary file. 

    Args:
        patientName (str): the name of the patient folder containing dicom 
            files (CT images) 
        organList (list): a list of organs that contours have been predicted for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        contoursList (list): a list containing the predicted contours, in the same 
            order as organList

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   

    patientPath = "Patient_Files/" + patientName
    #get the list of patient folders
    patientPath = os.path.join(path, patientPath)
    patientFolder = sorted(os.listdir(patientPath))
    if len(patientFolder) == 1 and os.path.isdir(os.path.join(patientPath, patientFolder[0])):
        patientPath = os.path.join(patientPath, patientFolder[0])
        patientFolder = os.listdir(patientPath)
    structFile = None

    for fileName in patientFolder:
        try:
            filePath = os.path.join(patientPath, fileName)
            patientData = pydicom.dcmread(filePath)
        except:
            continue
        modality = patientData[0x0008,0x0060].value 
        if "STRUCT" in modality:
            structFile = fileName

    #create a new struct file if there wasn't one provided
    # if structFile is None:
    rtStruct = RTStructBuilder.create_new(dicom_series_path = patientPath)
    rand_num = random.randrange(100000,999999)
    newStructFile = "ORG_STRUCT" + str(rand_num)
    # else:
    #     structPath = os.path.join(patientPath, structFile)
    #     #load existing RT Struct
    #     rtStruct = RTStructBuilder.create_from(dicom_series_path = patientPath, rt_struct_path = structPath)
    #     if "ORG" not in structFile:
    #         newStructFile =  "ORG_" + structFile.split(".dcm")[0] + "_1"
    #     else: newStructFile = structFile.split(".dcm")[0] + "_1"    

    #create a list of colors for the contours 
    colorList= [
        [255, 0, 255], #magenta
        [0, 235, 235], #teal
        [255, 255, 0], #yellow
        [255, 0, 0], #red
        [255, 175, 0], #orange
        [160, 32, 240], #purple
        [255, 140, 190], #pink
        [0, 0, 255], #blue
    ]

    for i, contours in enumerate(contoursList):
        #reshape contours so that it is compatible with dicom files
        newContours = []
        for slice in contours:
            newPoints = []
            for point in slice:
                newPoints.append(point[0])
                newPoints.append(point[1])
                newPoints.append(point[2])
            newContours.append(newPoints)

        contours = newContours

        #if there is already a contour under the name of the organ rename to "organ_1"
        ROIName = "ORG_" + organList[i]
        ROIName = ROIName[0:16]
        # for element in rtStruct.ds.StructureSetROISequence:
            # if str(element.ROIName).lower() == organList[i].lower():
            #     ROIName = organList[i] + "_1"

        #assign the contour a color
        if i < len(colorList):
            organColor = colorList[i]
        else: 
            organColor = colorList[i-len(colorList)]

        #make the body contour green by convention
        if organList[i].lower() == "body":
            organColor = [0,255,0] 

        #add the ROI
        rtStruct.add_roi(contours = contours, color = organColor, name = ROIName)

    #save the ROI to a new struct file
    rtStruct.save(str(os.path.join(patientPath, newStructFile)))
