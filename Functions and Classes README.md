## Functions 
All Organogenesis functions that do not belong to a class are detailed below. Functions in the files rtstruct.py, rtstruct_builder.py, rtuitls.py, and ds_helper.py are not included. Documentation for these functions is available at https://github.com/qurit/rt-utils. 

### AddZToContours(contours, zValues)

Adds the z value to each point in contours as it only contains x and y values.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (only x and y values) at a specific 
      z value

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

__________________________________________________________________________________________________________________________________________________________________

### AllowedToMatch(s1, s2)

Determines whether or not s1 and s2 are allowed to match based on if they both contain the correct substrings.  

**Arguments:**

      s1 (str): first string to determine match

      s2 (str): second string to determine match

**Returns:**

      allowed (bool): True if the strings are allowed to match, false otherwise

__________________________________________________________________________________________________________________________________________________________________

### BestThreshold(organ, path, modelType, testSize=500)

Determines the best threshold for predicting contours based on the F score. Displays graphs of the accuracy, false positives, false negatives, and F score vs threshold. Also saves these stats to the model folder.  

**Arguments:**

      organ (str): the organ to get the best threshold for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      modelType (str): the type of model

      testSize (str, int): the number of images to test with for each threshold

**Returns:**
      bestThreshold (float): the best threshold for predicting contours based on F score

__________________________________________________________________________________________________________________________________________________________________

### ClosestPoint(point, contours)

Finds the closest point to the given point within the contours list.  

**Arguments:**

      point (list): the point [x,y]

      contours (list): a list of lists. Each item is a list of the predicted contour points (only x and y values) at a specific 
      z value

**Returns:**

      closestPoint (list): the closest point to the given point in the contours list [x,y]

__________________________________________________________________________________________________________________________________________________________________

### Crop(tensor, targetTensor)

Crops tensor to the size of targetTensor.  

**Arguments:**

      tensor (2D tensor): the tensor to be cropped

      targetTensor (int): the tensor of the desired size

**Returns:**

      tensor: the cropped tensor

__________________________________________________________________________________________________________________________________________________________________

### DICOM_to_Image_Coordinates(IPP, pixelSpacing, contours)

Converts the coordinates in contours from real world (dicom) coordinates to pixel coordinates.  

**Arguments:**

      IPP (list): image position patient, the real world (dicom) coordinate of the top left hand corner of the CT image [x,y,z]

      pixelSpacing (list): the real world (dicom) distance between the centre of each pixel [x distance, y distance] in mm

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value [x,y,z] in real world (dicom) coordinates

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value [x,y,z] in  pixel coordinates

__________________________________________________________________________________________________________________________________________________________________

### DoubleConv(inC, outC)

Performs two successive 3x3 convolutions, each followed by a batch normalization, and a ReLU.  

**Arguments:**

      inC (int): input channels, the number of channels of the inputted tensor

      outC (int): output channels, the number of channels of the outputted tensor

**Returns:**

      conv (2D tensor): the tensor that has been modified

__________________________________________________________________________________________________________________________________________________________________

### Export_To_ONNX(organ)

‌Takes a PyTorch model and converts it to an Open Neural Network Exchange (ONNX) model for operability with other programming languages. The new model is saved into the Models folder.

**Arguments:**

     organ (str): the name of the organ for which a PyTorch model is to be converted to an ONNX model
    modelType (str): specifies whether a UNet or MultiResUnet model is to be converted

__________________________________________________________________________________________________________________________________________________________________

### FScore(organ, path, threshold, modelType)

Computes the F score, accuracy, precision, and recall for a model on a given organ. Uses the validation data.  

**Arguments:**

      organ (str): the organ to get the F score for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float) : the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

**Returns:**

      f_Score (float): (2 * precision * recall) / (precision + recall)

      recall (float): truePositives / (truePositives + falseNegatives)

      precision (float): truePositives / (truePositives + falsePositives)

      accuracy (float): (truePositives + trueNegatives) / allPixels

__________________________________________________________________________________________________________________________________________________________________

### FilterBackground(image, threshold)

Creates a binary mask of the background of the image. Pixels below the threshold are assigned a 1 (are not the organ), and 0 (are the organ) otherwise. This is the opposite of FilterContour.  

**Arguments:**

      image (2D numpy array): an array of pixels with values between 0 and 1

      threshold (float): the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

**Returns:**

      filteredImage (2D numpy array): the binary mask of the background

__________________________________________________________________________________________________________________________________________________________________

### FilterContour(image, threshold)

Creates a binary mask. Pixels above the threshold are assigned a 1 (are the organ), and 0 (are not the organ) otherwise. This is the opposite of FilterBackground.  

**Arguments:**

      image (2D numpy array): an array of pixels with values between 0 and 1

      threshold (float): the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

**Returns:**

      filteredImage (2D numpy array): the binary mask

__________________________________________________________________________________________________________________________________________________________________

### FilterIslands(slices)

Removes islands by recursively taking the largest chunk in the list. Lists are separated by 5 or more slices.  

**Arguments:**

      slices (list): list of indices for slices with at least one point

**Returns:**

      slices (list): the list of indices with islands removed

__________________________________________________________________________________________________________________________________________________________________

### FindStructure(metadata, organ, invalidStructures = [])

Finds the matching structure to a given organ in a patient's dicom file metadata.  

**Arguments:**

      metadata (DataElement): the data contained in the StructureSetROISequence of the patient's dicom file

      organ (str): the organ to find the matching structure for

      invaidStructures (list): a list of structures that the matching structure cannot be, defaults to an empty list

**Returns:**

      str, int: the matching structure's name in the metadata, the matching structure's ROI number in the metadata. 
      Returns "", 1111 if no matching structure is found in the metadata

__________________________________________________________________________________________________________________________________________________________________

### FixContours(orig_contours)

Creates additional interpolated contour slices if there are missing slice or if the slice has less than 4 points.  

**Arguments:**

      orig_contours (list): a list of lists. Each item is a list of the predicted contour points as 1D arrays [x,y] at a 
      specific z value

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points [x,y] at a specific 
      z value

__________________________________________________________________________________________________________________________________________________________________

### GetContours(organ, patientName, path, threshold, modelType, withReal = True, tryLoad=True, plot=True)

Uses a pre trained model to predict contours for a given organ. Saves the contours to the Predictions_Patients folder in a binary file.  

**Arguments:**

      organ (str): the organ to predict contours for

      patientName (str): the name of the patient folder containing dicom files (CT images) to predict contours for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float): the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

      withReal (bool): True to get existing contours from dicom file, defaults to True

      tryLoad (bool): True to try to load previously processed contours to save time, defaults to True plot 
      (bool) True to plot predicted contours, defaults to True

**Returns:**

      contoursList (list): list of contour points from the predicted contours

      existingContoursList (list): list of contour points from the existing contours

      contours (list): a list of lists. Each item is a list of the predicted contour points [x,y,z] at a specific 
      z value

      existingcontours (list): a list of lists. Each item is a list of the existing contour points [x,y,z] at a specific 
      z value

__________________________________________________________________________________________________________________________________________________________________

### GetDICOMContours(patientName, organ, path)

Gets the contours of a given organ from a given patient's dicom file.  

**Arguments:**

      patientName (str): the name of the patient folder containing the dicom files (CT images) to get the contours 
      of organ (str) the organ to get contours of

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

__________________________________________________________________________________________________________________________________________________________________

### GetEvalData(organ, path, threshold, modelType)

Creates a text file containing the hyperparameters of the model, the F score data, and the 95th percentile Haussdorff distance.  

**Arguments:**

      organ (str): the organ to get evaluation data for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float): the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

__________________________________________________________________________________________________________________________________________________________________

### GetMasks(organ, patientName, path, threshold, modelType)

Gets the existing and predicted masks of a given organ for a given patient.  

**Arguments:**

      organ (str): the organ to predict contours for

      patientName (str): the name of the patient folder containing dicom files (CT images) to get masks for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float) : the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

**Returns:**

      predictionsArray (3D numpy array): predicted masks of the organ for a given patient

      exisitingArray (3D numpy array): existing masks of the organ for a given patient

__________________________________________________________________________________________________________________________________________________________________

### GetMultipleContours(organList, patientName, path, thresholdList, modelType, withReal = True, tryLoad=True, plot=True, save=True) 

Calls the GetContours function to predict contours for each organ in organList using a pretrained model and then plots all of the predicted contours.  

**Arguments:**

      organList (list): a list of organs to predict contours for

      patientName (str): the name of the patient folder containing dicom files (CT images) to predict contours for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      thresholdList (list): a list of floats containing the cutoff for deciding if a pixel is an organ (assigned a 1) or not 
      (assigned a 0). Corresponding thresholds must be in the same order as organList

      modelType (str): the type of model

      withReal (bool): True to get existing contours from dicom file, defaults to True

      tryLoad (bool): True to try to load previously processed contours to save time, defaults to True 

      plot (bool) True to plot predicted contours, defaults to True
      save (bool): True to save predicted contours to a dicom file, defaults to True


**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points [x,y,z] at a specific 
      z value

      existingcontours (list): a list of lists. Each item is a list of the existing contour points [x,y,z] at a specific 
      z value

__________________________________________________________________________________________________________________________________________________________________

### GetOriginalContours(organ, patientName, path)

Gets the original contours from a given patient's dicom file for a given organ.  

**Arguments:**

      organ (str): the organ to get the original contours of

      patientName (str): the name of the patient folder containing dicom files (CT images) to get contours from

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

**Returns:**

      existingcontours (list): a list of lists. Each item is a list of the existing contour points [x,y,z] at a specific 
      z value

      existingContoursList (list): a list of contour points from the existing contours

__________________________________________________________________________________________________________________________________________________________________

### GetPointsAtZValue(contours, zValue) 

Gets all the points [x,y] at a given z value in a contour  

**Arguments:**

      contours (list): a list of lists. Each item is a list of contour points (x, y, and z values) at a specific 
      z value

      zValue (float): the z value of the slice to get the points for

**Returns:**

      pointsList (list): a list of all the points [x,y] at the given z value

__________________________________________________________________________________________________________________________________________________________________

### GetPredictionCTs(patientName, path)

Processes CT images for a given patient from a dicom file and saves them for the purpose of predicting.  

**Arguments:**

      patientName (str): the name of the patient folder to process data for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

__________________________________________________________________________________________________________________________________________________________________

### GetTrainingData(filesFolder, organ, path, sortData=False, preSorted=False)

Processes the dicom file data and saves it. CT images, masks, and z values are saved for the training and validation data. 10% of patients with good contours are saved to the validation data set. CT images are saved for the testing data. Contour bools (whether or not there is an existing contour for that slice) are saved for all data types.  

**Arguments:**

      filesFolder (str): the directory path to the folder with patient files

      organ (str): the organ to get data for

      preSorted(bool): True to use presorted good/bad/no contour lists, False to display contours for each 
      patient and sort manually

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

__________________________________________________________________________________________________________________________________________________________________

### GetZValues(contours)

Gets all of the z values in a contour.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of contour points (x, y, and z values) at a specific 
      z value

**Returns:**

      zValueList (list): a list of all the z values in a contour

__________________________________________________________________________________________________________________________________________________________________

### HaussdorffDistance(organ, path, threshold, modelType)

Determines the 95th percentile Haussdorff distance for a model with a given organ.  

**Arguments:**

      organ (str): the organ to get the Haussdorff distance for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float): the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

**Returns:**

      float: 95th percentile Haussdorff distance

__________________________________________________________________________________________________________________________________________________________________

### ImageUpsizer(array, factor)

Supersizes an array by the factor given.  

**Arguments:**

      array (2D numpy array): an array to be supersized

      factor (int): the amount to supersize the array by

**Returns:**

      newArray (2D numpy array): the supersized array
__________________________________________________________________________________________________________________________________________________________________

### Infer_From_ONNX(organ, patient)

Predicts organ masks using an ONNX model and plots slices for viewing. The ONNX model must be in the Models directory with the name “Model_{organ}.” This function is limited to viewing and cannot be used for creating contour lists or DICOM files.

**Arguments:**

      organ (str): organ for which binary masks are to be predicted using an ONNX model

      patient (str): The name of the patient within the Patient_Files directory which is to have organ masks predicted

___________________________________________________________________________________________________________________________________________________________

### InterpolateContour(contours1, contours2, distance)

Creates a linearly interpolated contour slice between contours1 and contours2.  

**Arguments:**

      contours1 (list): the first slice to be interpolated with. A list of [x,y] coordinates

      contours2 (list): the second slice to be interpolated with A list of [x,y] coordinates

      distance (int): the z distance between contours1 and contours2

**Returns:**

      newContour (list): the interpolated slice. A list of [x,y] coordinates

__________________________________________________________________________________________________________________________________________________________________

### InterpolatePoint(point1, point2, totalDistance, distance = 1)

Perfoms linear interpolation between point1 and point2.  

**Arguments:**

      point1 (list): the first point to be interpolated [x,y]

      point2 (list): the second point to be interpolated [x,y]

      totalDistance (int, float): the z distance between point 1 and point 2

      distance (int, float): the z distance between point 1 and the z value of the interpolated point

**Returns:**

      list: the interpolated point [x,y]

__________________________________________________________________________________________________________________________________________________________________

### InterpolateSlices(contours, patientName, organ, path, sliceThickness)

Interpolates slices with an unreasonable area, an unreasonable number of points, or with missing slices using the closest slice above and below.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

      patientName (str): the name of the patient folder to interpolate contours for

      organ (str): the organ to interpolate contours for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      sliceThickness (float): the distance between each CT slice in mm

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted and interpolated contour points (x, y, and z values) 
      at a specific z value
__________________________________________________________________________________________________________________________________________________________________

### LongestSubstring(s1,s2)

Finds the length of the longest substring that is in both s1 and s2.  

**Arguments:**

      s1 (str): the first string to find the longest substring in

      s2 (str): the second string to find the longest substring in

**Returns:**

      longest (int): the length of the longest substring that is in s1 and s2

__________________________________________________________________________________________________________________________________________________________________

### MaskOnImage(image, mask)

takes a normalized CT image and a binary mask, and creates a normalized image with the mask on the CT.  

**Arguments:**

      image (2D numpy array): the CT image which the mask image corresponds to.

      mask (2D numpy array): the binary mask which is to be placed on top of the CT image.

**Returns:**

      (2D numpy array): a normalized image of the mask placed on top of the CT image, with the binary mask 
      pixel value being 1.2x the maximum CT pixel value.

__________________________________________________________________________________________________________________________________________________________________

### MaskToContour(image)

Creates a contour from a mask.  

**Arguments:**

      image (2D numpy array): the mask to create a contour from

**Returns:**

      edges (2D array): an array of the same dimensions as image with values 0 (pixel is not an edge) or 255 (pixel is an edge)

      contours (ndarray): an array of the contour points for image

      combinedContours (ndarray): an array of the contour points for image combined from multiple contours

__________________________________________________________________________________________________________________________________________________________________

### MaxGap(slices)

Finds the largest separation of adjacent integers in a list of integers.  

**Arguments:**

      slices (list): list of indices for slices with at least one point

**Returns:**

      maxGap (int): the largest difference between adjacent indices in slices

      maxGapIndex (int): the index at which the largest gap occurs

__________________________________________________________________________________________________________________________________________________________________

### MissingSlices(contours, sliceThickness)

Determines which slices in the predicted contour are missing. Contours are expected to be continuous from the lowest z value to the highest.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

      sliceThickness (float): the distance between each CT slice in mm

**Returns:**

      missingZValues (list): a list of z values for the missing slices

__________________________________________________________________________________________________________________________________________________________________

### NormalizeImage(image)

Normalizes an image between 0 and 1.  

**Arguments:**

      image (ndarray): the image to be normalized

**Returns:**

      ndarray: the normalized image

__________________________________________________________________________________________________________________________________________________________________

### PercentStats(organ, path)

Saves lists of the area and the number of contour points at each percentage through a contour. Saves the max, min, and average area and number of contour points at each percentage through the contour. Gets the data from patient data sorted into the good contours list.  

**Arguments:**

      organ (str): the organ to get evaluation data for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

__________________________________________________________________________________________________________________________________________________________________

### PixelToContourCoordinates(contours, ipp, zValues, pixelSpacing, sliceThickness)

Converts the coordinates in contours from pixel coordinates to real world (dicom) coordinates.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value [x,y] in pixel coordinates

      ipp (list): image position patient, the real world (dicom) coordinate of the top left hand corner of the CT image [x,y,z]

      zValues (list): a list of the z values for the slices in contours

      pixelSpacing (list): the real world (dicom) distance between the centre of each pixel [x distance, y distance] in mm

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value [x,y,z] in real world (dicom) coordinates

      sliceThickness (float): the distance between each CT slice in mm

**Returns:**

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value [x,y,z] in real world (dicom) coordinates

__________________________________________________________________________________________________________________________________________________________________

### PlotPatientContours(contours, existingContours)

Plots the contours provided.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points at a specific 
      z value

      existingContours (list): a list of lists. Each item is a list of the existing contour points at a specific 
      z value.

__________________________________________________________________________________________________________________________________________________________________

### Process(prediction, threshold)

Performs a sigmoid and filters the prediction to a given threshold.  

**Arguments:**

      prediction (2D numpy array): an array of the predicted pixel values

      threshold (float) : the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

__________________________________________________________________________________________________________________________________________________________________

### sigmoid(z)

Performs a sigmoid function on z.  

**Arguments:**

      z (ndarray): the array to be modified

**Returns:**

      ndarray: the array after a sigmoid function has been applied

__________________________________________________________________________________________________________________________________________________________________

### StringDistance(s1, s2)

returns the Damerau-Levenshtein distance between two strings  

**Arguments:**

      s1 (string): string one which is to be compared with string 2.

      s2 (string): string two which is to be compared with string 1.

**Returns:**

      (int): the Damerau Levenshtein distance between s1 and s2, which indicates how different the two 
      strings are in terms of the amount of deletion, insertion, substitution, and transposition 
      operations required to equate the two.

__________________________________________________________________________________________________________________________________________________________________

### TestPlot(organ, path, threshold, modelType)

Plots 2D CTs with both manually drawn and predicted masks for visual comparison.  

**Arguments:**

      organ (str): the organ to plot masks for

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      threshold (float) : the cutoff for deciding if a pixel is an organ (assigned a 1) or not (assigned a 0)

      modelType (str): the type of model

__________________________________________________________________________________________________________________________________________________________________

### Train(organ,numEpochs,lr, path, processData, loadModel, modelType, sortData=False, preSorted=False, dataAugmentation=False)

Trains a model for predicting contours on a given organ. Saves the model and loss history after each epoch. Stops training after a given number of epochs or when the validation loss has decreased by less than 0.001 for 4 consecutive epochs.  

**Arguments:**

      organ (str): the organ to train the model on

      numEpochs (int, str): the number of epochs to train for

      lr (float, str): the learning rate to train with

      path (str): the path to the directory containing organogenesis folders (Models, Processed_Data, etc.)

      processData (bool): True to process the data into training/testing/ validation folders, False if they are 
      already processed

      loadModel (bool): True to load a model and continue training, False to train a new model

      modelType (str): the type of model to be used in training

      sortData (bool): True to visually inspect contours for quality assurance, False to process data without 
      looking at contours

      preSorted(bool): True to use presorted good/bad/no contour lists, False to display contours for each 
      patient and sort manually

      dataAugmentation (bool): True to turn on data augmentation for training, False to use non-augmented CT images.

__________________________________________________________________________________________________________________________________________________________________

### UnreasonableArea(contours, organ , path)

Determines which slices in the predicted contour have an unreasonable area. Unreasonable area is defined as less than the minimum area or more than the maximum area at a percentage through a contour. Based on the stats from the PercentStats function.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

      organ (str): the organ that contours were predicted for

      path (str): the path to the directory containing organogenesis folders

**Returns:**

      unreasonableArea (list): a list of z values for slices with an unreasonable area

__________________________________________________________________________________________________________________________________________________________________

### UnreasonableNumPoints(contours, organ, path)

Determines which slices in the predicted contour have an unreasonable number of points. An unreasonable number of points is defined as less than the minimum number of points at a specific percentage through the contour multiplied by a factor. Based on the stats from the PercentStats function and factors are organ specific.  

**Arguments:**

      contours (list): a list of lists. Each item is a list of the predicted contour points (x, y, and z values) at a specific 
      z value

      organ (str): the organ that contours were predicted for

      path (str): the path to the directory containing organogenesis folders

**Returns:**

      unreasonableNumPoints (list): a list of z values for slices with an unreasonable number of points

__________________________________________________________________________________________________________________________________________________________________

### Validate(organ, model)

Computes the average loss of the model on the validation data set.  

**Arguments:**

      organ (str): the organ to train the model on

      model (Module): the model to find the validation loss on

**Returns:**

      float: the average loss on the entire validation data set

## Classes 
The five Organogenesis classes are detailed below. Classes in the files rtstruct.py, rtstruct_builder.py, rtuitls.py, and ds_helper.py are not included. Documentation for these classes is available at https://github.com/qurit/rt-utils. 

### 1. ContourPredictionError(Exception)
Exception raised if there are no contour points in a contour. ContourPredictionError's method is detailed below.

### \_\_init__(self, message = "Contour Prediction Error. No contour points were predicted by the model. Please check that you are using the right threshold and try again."):
Instantiation method of ContourPredictionError. 

**Arguments:**

    message (string): the message to be printed if the contour 
                prediction error is raised 

### 2. CTDataset(Dataset)
A Dataset that stores CT images and their masks. CTDatset's methods are detailed below. 

### \_\_getitem__(self, index)

Gets the image and mask in the dataset at a given index. Performs data augmentation if it is being used.  

**Arguments:**

      index (int): the index of the image and mask to return

**Returns:**

      image (3D tensor): a normalized tensor of the image (num channels x height x width)

      mask (3D tensor): a tensor of the mask (num channels x height x width)
__________________________________________________________________________________________________________________________________________________________________

### \_\_init__(self, dataFiles, root_dir, transform)

Instantiation method for CTDataset.  

**Arguments:**

      dataFiles (list): a list containing the names of the files with CT data (images and masks)

      root_dir (str): the path to the directory containing the dataFiles

      transform (Compose, None): the transform(s) to be used for data augmentation

__________________________________________________________________________________________________________________________________________________________________

### \_\_len__(self)

Gets how many images there are in the dataset.

**Returns:** 

      int: the number of images in the dataset

__________________________________________________________________________________________________________________________________________________________________
### NormalizeImage(self, image)

Normalizes an image between 0 and 1.  

**Arguments:**

      image (2D numpy array): the image to be normalized

**Returns:**

      2D numpy array: the normalized image

### 3. MultiResBlock
A class for the MultiResBlock which is a part of MultiResUNet. 

### forward(self, image)

Performs the steps of the MultiResBlock.  

**Arguments:**

      image (2D tensor): the tensor to go through the MultiResBlock

**Returns:**
      final (2D tensor) the modified tensor 

__________________________________________________________________________________________________________________________________________________________________

### \_\_init__(self,inC, outC, alpha)

Instantiation method for the MultiResBlock class.  

**Arguments:**

      inC (int): input channels, the number of channels of the inputted tensor

      outC (int): output channels, the number of channels of the outputted tensor

      alpha (float): scalar coefficient which can be adjusted to change the number of parameters in the model

### 4. MultiResUNet
A class for the convolutional neural network MultiResUNet. MultiResUNet's methods are detailed below.

### forward(self, image)

Performs the U architecture (encoder, decoder, and Res paths) of MultiResUNet.  

**Arguments:**

      image (2D tensor): the input image

**Returns:**

      x (2D tensor): the segmentation mask

__________________________________________________________________________________________________________________________________________________________________

### \_\_init__(self)

Instantiation method for MultiResUNet.
__________________________________________________________________________________________________________________________________________________________________

### trainingStep(self, x, y)

Performs a training step.  

**Arguments:**

      x (2D tensor): the CT image

      y (2D tensor): the ground truth mask

**Returns:**

      loss (float): the binary cross entropy loss of the model using training data

__________________________________________________________________________________________________________________________________________________________________

### validationStep(self, x, y)

Performs a validation step.  

**Arguments:**

      x (2D tensor): the CT image

      y (2D tensor): the ground truth mask

**Returns:**

      loss (float): the binary cross entropy loss of the model using validation data

### 5. UNet(nn.Module)
A class for the convolutional neural network UNet. UNet's methods are detailed below. 

### forward(self, image)

Performs the U architecture (encoder and decoder) of UNet.  

**Arguments:**

      image (2D tensor): the input image

**Returns:**

      x (2D tensor): the segmentation mask

__________________________________________________________________________________________________________________________________________________________________

### \_\_init__(self)
Instantiation method for UNet. 

__________________________________________________________________________________________________________________________________________________________________

### trainingStep(self, x, y)

Performs a training step.  

**Arguments:**

      x (2D tensor): the CT image

      y (2D tensor): the ground truth mask

**Returns:**

      loss (float): the binary cross entropy loss of the model using training data

__________________________________________________________________________________________________________________________________________________________________

### validationStep(self, x, y)

Performs a validation step.  

**Arguments:**

      x (2D tensor): the CT image

      y (2D tensor): the ground truth mask

**Returns:**

      loss (float): the binary cross entropy loss of the model using validation data


