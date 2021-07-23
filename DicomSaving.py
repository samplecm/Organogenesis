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

def SaveToDICOM(patientName, organ, path, contours):
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   

    patientPath = "Patient_Files/" + patientName
    #get the list of patient folders
    patientPath = os.path.join(path, patientPath)
    patientFolder = sorted(os.listdir(patientPath))

    for fileName in patientFolder:
        if "STRUCT" in fileName:
            structFile = fileName  

    structPath = os.path.join(patientPath, structFile)

    #load existing RT Struct
    rtStruct = RTStructBuilder.create_from(dicom_series_path = patientPath, rt_struct_path = structPath)

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
    ROIName = organ
    for element in rtStruct.ds.StructureSetROISequence:
        if str(element.ROIName).lower() == organ.lower():
            ROIName = organ + "_1"

    #add the ROI
    rtStruct.add_roi(contours = contours, color=[0, 255, 0], name=ROIName)

    #save the ROI to a new struct file
    newStructFile = structFile.split(".")[0] + "_1"
    rtStruct.save(str(os.path.join(patientPath, newStructFile)))
