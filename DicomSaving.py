import numpy as np 
import pydicom
from pydicom import dcmread
import pickle
import os
import glob
import pathlib
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw
from operator import itemgetter
import matplotlib.pyplot as plt
import Preprocessing
import open3d as o3d
import random
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from rt_utils import RTStruct
from rt_utils.image_helper import get_contours_coords
from rt_utils.utils import ROIData, SOPClassUID
from rt_utils import ds_helper
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ImplicitVRLittleEndian

def AddContour():
    ds_helper.create_contour_sequence = NewCreateContourSequence

def NewCreateContourSequence(roi_data,series_data):
    """
    Iterate through each slice of the contour
    For each connected segment within a slice, create a contour
    """
    contour_coords = []
    contour_sequence = Sequence()
    for i, series_slice in enumerate(series_data):
        contour_slice = roi_data.mask[i]

        contour_slice = list(contour_slice)


        if len(contour_slice) == 0: #do not append contour info for the slice if there is none
            continue

        contour_coords.append(contour_slice)

        for contour_data in contour_coords:
            contour = ds_helper.create_contour(series_slice, contour_data)
            contour_sequence.append(contour)

    return contour_sequence

def NewValidateMask(self, mask):

    if len(self.series_data) != len(mask):
            raise RTStruct.ROIException(
                "Contour must have the save number of layers (In the 3rd dimension) as input series. " +
                f"Expected {len(self.series_data)}, got {len(mask)}")

    return True

def SaveToDICOM(organ, patientName, path, contours):
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

    # Load existing RT Struct. Requires the series path and existing RT Struct path

    rtstruct = RTStructBuilder.create_from(dicom_series_path = patientPath, rt_struct_path = structPath)

    ds_helper.create_contour_sequence = NewCreateContourSequence
    RTStruct.validate_mask = NewValidateMask

    #Add ROI. This is the same as the above example.
    
    rtstruct.add_roi(mask = contours, color=[255, 0, 255], name="RT-Utils ROI!")


    rtstruct.save('new-rt-struct')


    print("yep")