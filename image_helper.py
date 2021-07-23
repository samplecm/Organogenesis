import os
from typing import List
from enum import IntEnum

import cv2 as cv
import numpy as np
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from rtutils import ROIData, SOPClassUID


def load_sorted_image_series(dicom_series_path: str):
    """
    File contains helper methods for loading / formatting DICOM images and contours
    """

    series_data = load_dcm_images_from_path(dicom_series_path)

    if len(series_data) == 0:
        raise Exception("No DICOM Images found in input path")

    # Sort slices in ascending order
    series_data.sort(key=lambda ds: ds.ImagePositionPatient[2], reverse=False)

    return series_data


def load_dcm_images_from_path(dicom_series_path: str) -> List[Dataset]:
    series_data = []
    for root, _, files in os.walk(dicom_series_path):
        for file in files:
            try:
                ds = dcmread(os.path.join(root, file))
                if hasattr(ds, 'pixel_array'):
                    series_data.append(ds)

            except Exception:
                # Not a valid DICOM file
                continue

    return series_data


def get_contours_coords(mask_slice: np.ndarray, series_slice: Dataset, roi_data: ROIData):
    # Create pin hole mask if specified
    if roi_data.use_pin_hole:
        mask_slice = create_pin_hole_mask(mask_slice, roi_data.approximate_contours)

    # Get contours from mask
    contours, _ = find_mask_contours(mask_slice, roi_data.approximate_contours)
    validate_contours(contours)
    
    # Format for DICOM
    formatted_contours = []
    for contour in contours:
        contour = np.array(contour) # Type cannot be a list
        translated_contour = translate_contour_to_data_coordinants(contour, series_slice)
        dicom_formatted_contour = format_contour_for_dicom(translated_contour, series_slice)
        formatted_contours.append(dicom_formatted_contour)

    return formatted_contours 


def find_mask_contours(mask: np.ndarray, approximate_contours: bool):
    approximation_method = cv.CHAIN_APPROX_SIMPLE if approximate_contours else cv.CHAIN_APPROX_NONE 
    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, approximation_method)
    # Format extra array out of data
    for i, contour in enumerate(contours):
        contours[i] = [[pos[0][0], pos[0][1]] for pos in contour]
    hierarchy = hierarchy[0] # Format extra array out of data

    return contours, hierarchy



def create_pin_hole_mask(mask: np.ndarray, approximate_contours: bool):
    """
    Creates masks with pin holes added to contour regions with holes.
    This is done so that a given region can be represented by a single contour.
    """

    contours, hierarchy = find_mask_contours(mask, approximate_contours)
    pin_hole_mask = mask.copy()

    # Iterate through the hierarchy, for child nodes, draw a line upwards from the first point
    for i, array in enumerate(hierarchy):
        parent_contour_index = array[Hierarchy.parent_node]
        if parent_contour_index == -1: continue # Contour is not a child

        child_contour = contours[i]
        
        line_start = tuple(child_contour[0])

        pin_hole_mask = draw_line_upwards_from_point(pin_hole_mask, line_start, fill_value=0)
    return pin_hole_mask


def draw_line_upwards_from_point(mask: np.ndarray, start, fill_value: int) -> np.ndarray:
    line_width = 2
    end = (start[0], start[1] - 1)
    mask = mask.astype(np.uint8) # Type that OpenCV expects
    # Draw one point at a time until we hit a point that already has the desired value
    while mask[end] != fill_value:
        cv.line(mask, start, end, fill_value, line_width)

        # Update start and end to the next positions
        start = end
        end = (start[0], start[1] - line_width)
    return mask.astype(bool)


def validate_contours(contours: list):
    if len(contours) == 0:
        raise Exception("Unable to find contour in non empty mask, please check your mask formatting")


def translate_contour_to_data_coordinants(contour, series_slice: Dataset):
    offset = series_slice.ImagePositionPatient
    spacing = series_slice.PixelSpacing
    contour[:, 0] = (contour[:, 0]) * spacing[0] + offset[0]
    contour[:, 1] = (contour[:, 1]) * spacing[1] + offset[1]
    return contour


def translate_contour_to_pixel_coordinants(contour, series_slice: Dataset):
    offset = series_slice.ImagePositionPatient
    spacing = series_slice.PixelSpacing
    contour[:, 0] = (contour[:, 0] - offset[0]) / spacing[0]
    contour[:, 1] = (contour[:, 1] - + offset[1]) / spacing[1] 

    return contour


def format_contour_for_dicom(contour, series_slice: Dataset):
    # DICOM uses a 1d array of x, y, z coords
    z_indicies = np.ones((contour.shape[0], 1)) * series_slice.ImagePositionPatient[2]
    contour = np.concatenate((contour, z_indicies), axis = 1)
    contour = np.ravel(contour)
    contour = contour.tolist()
    return contour


def create_series_mask_from_contour_sequence(series_data, contour_sequence: Sequence):
    mask = create_empty_series_mask(series_data)

    # Iterate through each slice of the series, If it is a part of the contour, add the contour mask
    for i, series_slice in enumerate(series_data):
        slice_contour_data = get_slice_contour_data(series_slice, contour_sequence)
        if len(slice_contour_data):
            mask[:, :, i] = get_slice_mask_from_slice_contour_data(series_slice, slice_contour_data)
    return mask


def get_slice_contour_data(series_slice: Dataset, contour_sequence: Sequence):
    slice_contour_data = []
    
    # Traverse through sequence data and get all contour data pertaining to the given slice
    for contour in contour_sequence:
        for contour_image in contour.ContourImageSequence:
            if contour_image.ReferencedSOPInstanceUID == series_slice.SOPInstanceUID:
                slice_contour_data.append(contour.ContourData)

    return slice_contour_data


def get_slice_mask_from_slice_contour_data(series_slice: Dataset, slice_contour_data):
    slice_mask = create_empty_slice_mask(series_slice)
    for contour_coords in slice_contour_data:    
        fill_mask = get_contour_fill_mask(series_slice, contour_coords)
        # Invert values in the region to be filled. This will create holes where needed if contours are stacked on top of each other
        slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])
    return slice_mask


def get_contour_fill_mask(series_slice: Dataset, contour_coords):
    # Format data
    reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
    translated_contour_data  = translate_contour_to_pixel_coordinants(reshaped_contour_data, series_slice)
    polygon = [np.array([translated_contour_data[:, :2]], dtype=np.int32)]

    # Create mask for the region. Fill with 1 for ROI
    fill_mask = create_empty_slice_mask(series_slice).astype(np.uint8)
    cv.fillPoly(img=fill_mask, pts=polygon, color=1)
    return fill_mask


def create_empty_series_mask(series_data):
    ref_dicom_image = series_data[0]
    mask_dims = (int(ref_dicom_image.Columns), int(ref_dicom_image.Rows), len(series_data))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


def create_empty_slice_mask(series_slice):
    mask_dims = (int(series_slice.Columns), int(series_slice.Rows))
    mask = np.zeros(mask_dims).astype(bool)
    return mask


class Hierarchy(IntEnum):
    """
    Enum class for what the positions in the OpenCV hierarchy array mean
    """

    next_node = 0
    previous_node = 1
    first_child = 2
    parent_node = 3