from typing import List, Union
import numpy as np
from pydicom.dataset import FileDataset
from rtutils import ROIData
import ds_helper

class RTStruct:
    """
    Wrapper class to facilitate appending and extracting ROI's within an RTStruct
    """

    def __init__(self, series_data, ds: FileDataset):
        self.series_data = series_data
        self.ds = ds
        self.frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[-1].FrameOfReferenceUID  # Use last strucitured set ROI

    def set_series_description(self, description: str):
        """
        Set the series description for the RTStruct dataset
        """

        self.ds.SeriesDescription = description

    def add_roi(
        self,
        contours: np.ndarray,
        color: Union[str, List[int]] = None,
        name: str = None,
        description: str = ''
        ):
        """
        Add a ROI to the rtstruct given contours for the ROI's at each slice
        Optionally input a color or name for the ROI
        """

        self.validate_contour(contours)
        roi_number = len(self.ds.StructureSetROISequence) + 1
        roi_number = self.validate_roiNumber(roi_number)
        roi_data = ROIData(
            contours,
            color,
            roi_number,
            name,
            self.frame_of_reference_uid,
            description
            )

        self.ds.ROIContourSequence.append(ds_helper.create_roi_contour(roi_data, self.series_data))
        self.ds.StructureSetROISequence.append(ds_helper.create_structure_set_roi(roi_data))
        self.ds.RTROIObservationsSequence.append(ds_helper.create_rtroi_observation(roi_data))

    def validate_contour(self, contours):

        if len(self.series_data) != len(contours):
                raise RTStruct.ROIException(
                    "Contour must have the save number of layers (In the 3rd dimension) as input series. " +
                    f"Expected {len(self.series_data)}, got {len(contours)}")

        return True

    def validate_roiNumber(self, roi_number):
        existingROINumbers = []
        for element in self.ds.ROIContourSequence:
            existingROINumbers.append(int(element.ReferencedROINumber))
        if roi_number in existingROINumbers:
            roi_number = max(existingROINumbers) + 1

        return roi_number

    def save(self, file_path: str):
        """
        Saves the RTStruct with the specified name / location
        Automatically adds '.dcm' as a suffix
        """

        # Add .dcm if needed
        file_path = file_path if file_path.endswith('.dcm') else file_path + '.dcm'

        try:
            file = open(file_path, 'w')
            # Opening worked, we should have a valid file_path
            print("Writing file to", file_path)
            self.ds.save_as(file_path)
            file.close()
        except OSError:
            raise Exception(f"Cannot write to file path '{file_path}'")

    class ROIException(Exception):
        """
        Exception class for invalid ROI masks
        """

        pass

    def get_roi_names(self) -> List[str]:
        """
        Returns a list of the names of all ROI within the RTStruct
        """

        if not self.ds.StructureSetROISequence:
            return []

        return [structure_roi.ROIName for structure_roi in self.ds.StructureSetROISequence]

    def get_roi_mask_by_name(self, name) -> np.ndarray:
        """
        Returns the 3D binary mask of the ROI with the given input name
        """

        for structure_roi in self.ds.StructureSetROISequence:
            if structure_roi.ROIName == name:
                contour_sequence = ds_helper.get_contour_sequence_by_roi_number(self.ds, structure_roi.ROINumber)
                return image_helper.create_series_mask_from_contour_sequence(self.series_data, contour_sequence)

        raise RTStruct.ROIException(f"ROI of name `{name}` does not exist in RTStruct")