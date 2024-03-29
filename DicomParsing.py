from logging import exception
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
import open3d as o3d
from scipy.ndimage.filters import gaussian_filter
import random
import shutil
from fastDamerauLevenshtein import damerauLevenshtein

    
#Want to have a function that checks for a structure in patients' structure sets, and obtains the contours
#if preSorted = True, that means that they are already sorted lists indicating whether the contour is good or bad (no need to plot) 
def GetTrainingData(filesFolder, organ, path, sortData=False, preSorted=False, cross_val=False, fold=None):
    """Processes the dicom file data and saves it. CT images, masks, and z values 
       are saved for the training and validation data. 10% of patients with good contours 
       are saved to the validation data set. CT images are saved for the testing data.
       Contour bools (whether or not there is an existing contour for that slice) are saved
       for all data types. 

    Args:
        filesFolder (str): the directory path to the folder with patient files
        organ (str): the organ to get data for
        preSorted(bool): True to use presorted good/bad/no contour lists, 
            False to display contours for each patient and sort manually
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)
        cross_val (bool) True if a cross validation is to be performed. If true, fold must be specified
        fold (list(int)): specifies 2 values, 1: the total folds used for cross validation, and 2: the current fold being calculated    

    """
    total_folds = None
    current_fold = None
    if cross_val==True:
        if fold==None:
            raise Exception("Cannot have cross_val == True if fold is unspecified")
        if type(fold) != list:
            raise Exception("Fold must be a list of two integers.")
        if len(fold) != 2:
            raise Exception("Fold must be a list of two integers.")
        total_folds = fold[0]
        current_fold = fold[1]    



    if fold!=None:
        if cross_val==False:
            raise exception("Warning: cross_val is false but a non-None fold has been specified.")       

    #get the list of patient folders
    patientFoldersRaw = sorted(os.listdir(filesFolder))
    #If the user has provided a path to patients files, then we don't want any of the subdirectory names created by the program to be included in this list, so filter them out 
    standardFolders = [
        "Loss History",
        "Models",
        "Patient_Files",
        "Predictions_Patients",
        "Processed_Data",
        "SavedImages",
        "FolderSetup.sh"
    ]
    patientFolders = []
    for folder in patientFoldersRaw:
        if folder not in standardFolders:
            patientFolders.append(folder)       

    #Create a dictionary for patients and their corresponding matched organ, and a list for unmatched patients. All the ones with matching organs can be used for training
    patientStructuresDict = {}
    unmatchedPatientsList = []
    if path != pathlib.Path(__file__).parent.absolute() and path is not None:
        preSorted = False #if path is given, assume it doesn't have the normal folder structure.

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()    

    #first delete processed data that is already in the organ directory, as it could interfere with the new sorting
    train_path = os.path.join(path, 'Processed_Data', organ)
    val_path = train_path + "_Val"
    train_files = os.listdir(train_path)
    val_files = os.listdir(val_path)
    for file in train_files:
        os.remove(os.path.join(train_path, file))
    for file in val_files:
        os.remove(os.path.join(val_path, file))



    if preSorted:
 
        with open(os.path.join(path, "Processed_Data/Sorted Contour Lists/" + organ + " Good Contours.txt"), "rb") as fp:
            goodContoursList = pickle.load(fp)

        for p in range(len(patientFolders)):
            if patientFolders[p] in goodContoursList:
                patient = sorted(glob.glob(os.path.join(filesFolder, patientFolders[p], "*")))
               
                structFiles = []
                for fileName in patient:
                    #get the modality: 
                    patientData = pydicom.dcmread(fileName)
                    modality = patientData[0x0008,0x0060].value 
                    if "STRUCT" in modality:
                        structFiles.append(fileName)  
                #structsMeta = dcmread(structFile).data_element("ROIContourSequence")
                structure, structureROINum, struct_idx = FindStructure(structFiles, organ)

                patientStructuresDict[str(patientFolders[p])] =  [structure, structureROINum]

        validationPatientsList = []

        validationPatientsListLength = int(len(goodContoursList)*0.1)

        for i in range(validationPatientsListLength):
            patientChoice = random.choice(goodContoursList)
            validationPatientsList.append(patientChoice)
            goodContoursList.remove(patientChoice)

        with open(os.path.join(path, "Processed_Data/Sorted Contour Lists/" + organ + " Bad or No Contours.txt"), "rb") as fp:
            unmatchedPatientsList = pickle.load(fp)

    else: 
    #Loop through the patients
        print(f"Searching for patients with {organ} contours.")
        for p in range(len(patientFolders)):
            patient = sorted(glob.glob(os.path.join(filesFolder, patientFolders[p], "*")))
            #get the RTSTRUCT dicom file and get patient 's CT scans: 
            structFiles = []
            for fileName in patient:
                #get the modality: 
                try:
                    patientData = pydicom.dcmread(fileName)
                except:
                    continue    
                modality = patientData[0x0008,0x0060].value 
                if "STRUCT" in modality:
                    structFiles.append(fileName)          
            structure, structureROINum, struct_idx = FindStructure(structFiles, organ)
            #add to dictionary if structureindex is not 1111 (error code)
            if structureROINum!= 1111: #error code from FindStructure
                patientStructuresDict[str(patientFolders[p])] =  [structure, structureROINum, struct_idx]
            else:
                unmatchedPatientsList.append(str(patientFolders[p]))   

    #now display these matched structures to the user:
    displayString = str(len(patientStructuresDict)) + "/" + str(len(patientFolders)) + " patients returned a structure for " + organ + "\n\n"
    if len(unmatchedPatientsList) > 0:
        if (preSorted == True):
            displayString += "Patients without a corresponding structure or a bad contour: \n"
        else: 
            displayString += "Patients without a corresponding structure: \n"
        displayString += "-------------------------------------------\n"
        i = 1
        for patient in unmatchedPatientsList:
            displayString += str(i) + ": " + patient + "\n"
            i += 1
    else:
        displayString += "Matching structures found for all patients\n"
    displayString += "------------------------------------------\n" 
    if (preSorted == True):
        displayString += "Patients with a good contour: \n"
    else: 
        displayString += "Patients with a corresponding structure: \n"
    displayString += "-------------------------------------------\n"
    i = 1
    for data in patientStructuresDict:
        displayString += str(i) + ". " + data + ": " + patientStructuresDict[data][0] + "\n"
        i += 1
    print(displayString) 
    #Unnecessary to wait after listing patients? 
    # while True:
    #     try:
    #         cont = input("Continue? (Y/N)")
    #         if (cont.lower() == "y"):
    #             break
    #         if cont.lower() == "n":
    #             quit()
    #     except KeyboardInterrupt:
    #         quit()
    #     except: pass   
    #Now loop over patients, saving image data for each if they have a structure
    for p in range(len(patientFolders)):
        if patientFolders[p] not in patientStructuresDict: #if doesn't have contour, add this CT to the test folder
            continue

        patient = sorted(glob.glob(os.path.join(filesFolder, patientFolders[p], "*")))
    
        #Now loop over patients for image data
        
        patient_CTs = []
        patient_Struct = []
        for fileName in patient:
            #get the modality: 
            try:
                patientData = pydicom.dcmread(fileName)
            except: continue    
            modality = patientData[0x0008, 0x0060].value 
            
            if modality == "CT":
                patient_CTs.append(fileName)
            elif "STRUCT" in modality:
                patient_Struct.append(fileName)  
        #iop and ipp are needed to relate the coordinates of structures to the CT images
        iop = dcmread(patient_CTs[0]).get("ImageOrientationPatient")
        ipp = dcmread(patient_CTs[0]).get("ImagePositionPatient")
        print([patientFolders[p], iop, ipp])
        #also need the pixel spacing
        pixelSpacing = dcmread(patient_CTs[0]).get("PixelSpacing")
        sliceThickness = dcmread(patient_CTs[0]).get("SliceThickness")
        #Now I want numpy arrays of all these CT images. I save these in a list, where each element is itself a list, with index 0: CT image, index 1: z-value for that slice
        CTs = []
        ssFactor = 1 #at the moment don't need to supersample the image. This is something to look into.. does increasing resolution increase the accuracy? SSfactor an integer greater than 1 will multiply the pixels in each direction by that number
        for item in pixelSpacing:
            item *= ssFactor
        for CTFile in patient_CTs:
            ctData = dcmread(CTFile)
            
            rescaleSlope = ctData[0x0028, 0x1053].value
            rescaleIntercept = ctData[0x0028, 0x1052].value
            array = np.array(ctData.pixel_array)
            array = array * rescaleSlope + rescaleIntercept
            resizedArray = ImageUpsizer(array , ssFactor)
            #Now normalize the image so its values are between 0 and 1
            #Anything greater than 2500 Hounsfield units is likely artifact, so cap here.
            resizedArray = np.clip(resizedArray, -1000, 2500)
            resizedArray = NormalizeImage(resizedArray)
            #images are now normalized in the dataset, this allows for data augmentation to be performed 
            #resizedArray = NormalizeImage(resizedArray)            
            CTs.append( [ resizedArray, dcmread(CTFile).data_element("ImagePositionPatient").value[2]])

        CTs.sort(key=lambda x:x[1]) #not necessarily in order, so sort according to z-slice.

        


        roiNumber = patientStructuresDict[patientFolders[p]][1]
        struct_idx = patientStructuresDict[patientFolders[p]][2]
        structsMeta = dcmread(patient_Struct[struct_idx]).data_element("ROIContourSequence")
        #print(dcmread(patient1_Struct[0]).data_element("ROIContourSequence")[0])
        #collect array of contour points for test plotting
        numContourPoints = 0 
        contours = []
        for contourInfo in structsMeta:
            if contourInfo.get("ReferencedROINumber") == roiNumber: #get the correct contour for the given organ
                for contoursequence in contourInfo.ContourSequence: 
                    contours.append(contoursequence.ContourData)
                    #But this is easier to work with if we convert from a 1d to a 2d list for contours ( [ [x1,y1,z1], [x2,y2,z2] ... ] )
                    tempContour = []
                    i = 0
                    while i < len(contours[-1]):
                        x = float(contours[-1][i])
                        y = float(contours[-1][i + 1])
                        z = float(contours[-1][i + 2])
                        tempContour.append([x, y, z ])
                        i += 3
                        if (numContourPoints > 0):
                            pointListy = np.vstack((pointListy, np.array([x,y,z])))
                        else:
                            pointListy = np.array([x,y,z])   
                        numContourPoints+=1       
                    contours[-1] = tempContour  
        #Right now I need the contour points in terms of the image pixel numbers so that they can be turned into an image for training:
        contourIndices = DICOM_to_Image_Coordinates(ipp, pixelSpacing, contours)
        #Now need to make images of the same size as CTs, with contour masks
        contourImages = []
        combinedImages = [] #mask on the CT
        backgroundImages = []
        contourOnPlane = np.zeros((len(CTs),1))
        xLen = np.size(CTs[0][0], axis = 0)
        yLen = np.size(CTs[0][0], axis = 1)
        for idx, CT in enumerate(CTs):            
            contourOnImage = False #keep track if a contour is on this image, if not just add a blank image.  

            #Now need to add the contour polygon mask to this image, if one exists on the current layer
            #loop over contours to check if z-value matches current CT.
            for contour in contourIndices:
                if int(round(contour[0][2], 2)*100) == int(round(CT[1], 2)*100):   #if the contour is on the current slice (match to 2 decimals)
                    contourOnPlane[idx] = 1
                    contourImage = Image.new('L', (xLen, yLen), 0 )#np.zeros((xLen, yLen))
                    backgroundImage = Image.new('L', (xLen, yLen), 1 )#np.zeros((xLen, yLen))
                    #combinedImage = Image.fromarray(CT[0])
                    contourPoints = []
                    #now add all contour points to contourPoints as Point objects
                    for pointList in contour:
                        contourPoints.append((int(pointList[0]), int(pointList[1])))
                    if len(contourPoints) > 3:
                        contourPolygon = Polygon(contourPoints)
                        ImageDraw.Draw(contourImage).polygon(contourPoints, outline= 1, fill = 1) #this now makes every pixel that the organ slice contains be 1 and all other pixels remain zero. This is a binary mask for training
                        #ImageDraw.Draw(combinedImage).polygon(contourPoints, outline= 1, fill = 1)
                        #ImageDraw.Draw(backgroundImage).polygon(contourPoints, outline= 0, fill = 0)
                        contourImage = np.array(contourImage)
                        contourImages.append(contourImage)
                        # combinedImage = np.array(combinedImage)
                        # combinedImages.append(combinedImage)
                        backgroundImage = np.array(backgroundImage)
                        backgroundImages.append(backgroundImage)
                        contourOnImage = True
                        break
            if not contourOnImage:
                #if no contour on that layer, add just zeros array
                contourImage = np.zeros((xLen,yLen))
                contourImages.append(contourImage)
                backgroundImage = np.ones((xLen, yLen))
                backgroundImages.append(backgroundImage)
                #combinedImages.append(CT[0])   

        #Now take these lists and combine them into one 4 dimensional numpy array (1st dim image type, second slice, third and 4th are pixels)
        #Need to create path strings for the saved file. 
        trainImagePath = "Processed_Data/" + organ + "/TrainData_" + patientFolders[p] 
        #trainContourBoolPath = "Processed_Data/" + organ + " bools/contourBool_" + patientFolders[p] 
        valImagePath = "Processed_Data/" + organ + "_Val/ValData_" + patientFolders[p]
        #valContourBoolPath = "Processed_Data/" + organ + " bools/contourBool_" + patientFolders[p] 
        #testImagePath = "Processed_Data/" + organ + "_Test/TestData_" + patientFolders[p] 
        #testContourBoolPath = "Processed_Data/" + organ + " bools/contourBool_" + patientFolders[p] 
        combinedData = np.zeros((2,len(CTs),xLen,yLen))
        slice_ZVals = [] #For storing the z value of CT slices
        for zSlice in range(len(CTs)):
            combinedData[0,zSlice,:,:] = CTs[zSlice][0]
            combinedData[1,zSlice,:,:] = contourImages[zSlice]
            slice_ZVals.append(CTs[zSlice][1]) #Add z value for the slice
            #combinedData[2,zSlice,:,:] = backgroundImages[zSlice]
        
        #Now also get the contour for plotting
        # body, bodyROINum= FindStructure(dcmread(structFile).data_element("StructureSetROISequence"), "body")
        # numContourPoints = 0 
        # bodyContours = []
        # sliceNum = 0
        # for contourInfo in structsMeta:
        #     if contourInfo.get("ReferencedROINumber") == bodyROINum:
        #         for contoursequence in contourInfo.ContourSequence: #take away 0!!
        #             sliceNum += 1
        #             if sliceNum % 2 == 0:
        #                 continue #take every second slice for imaging
        #             bodyContours.append(contoursequence.ContourData)
        #             #But this is easier to work with if we convert from a 1d to a 2d list for contours
        #             tempContour = []
        #             i = 0
        #             while i < len(bodyContours[-1]):
        #                 x = float(bodyContours[-1][i])
        #                 y = float(bodyContours[-1][i + 1])
        #                 z = float(bodyContours[-1][i + 2])
        #                 tempContour.append([x, y, z ])
        #                 i += 3
        #                 pointListy = np.vstack((pointListy, np.array([x,y,z])))
        #                 numContourPoints+=1       
        #             bodyContours[-1] = tempContour    



        #So some of the contoured organs are incomplete or just not good, and so you don't want to use for training. So what this is doing now is plotting every contour so that you can then decide whether or not to use it for training.
        #only need to plot the contours if they have not been sorted yet 
        if sortData == True and preSorted == False:
            print("Plotting")
            #Here we plot a 3d image of the pointcloud from the list of masks. 
            #First need to convert the masks to contours: 
            pointCloud = o3d.geometry.PointCloud()
            pointCloud.points = o3d.utility.Vector3dVector(pointListy[:,:3])
            o3d.visualization.draw_geometries([pointCloud])
            while True:
                try:
                    inp = input("Use for training? (Y/N)")
                    if (inp.lower() == "y"):
                        save = True
                        break
                    if inp.lower() == "n":
                        save = False
                        break
                except KeyboardInterrupt:
                    quit()
                except: pass  

        #always want to save the CTs and masks of the good contours if its been presorted already or sortdata is not selected.
        else: 
            save = True

        if cross_val==True:
            #need to define which files will be used for validation in current fold
            num_patients = len(patientStructuresDict.keys())
            vals_per_fold = int(num_patients / total_folds)
            val_indices = range(vals_per_fold*(current_fold - 1), vals_per_fold*(current_fold))#patient folder indices (p) that will be placed in validation set



        for zSlice in range(len(CTs)):    
            sliceText = str(zSlice)

            if save == True:             
                if zSlice < 10:
                    sliceText = "00" + str(zSlice)
                elif zSlice < 100:
                    sliceText = "0" + str(zSlice)    

                if preSorted == True: 

                    if patientFolders[p] not in validationPatientsList:
                        with open(os.path.join(path, str(trainImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)         
                        # with open(os.path.join(path, str(trainContourBoolPath)+ "_" + sliceText + ".txt"), "wb") as fp:
                        #     pickle.dump(contourOnPlane[zSlice], fp)
                    else: 
                        with open(os.path.join(path, str(valImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)         
                        # with open(os.path.join(path, str(valContourBoolPath + "_" + sliceText + ".txt")), "wb") as fp:
                        #     pickle.dump(contourOnPlane[zSlice], fp) 

                if cross_val==True:
                    if p in val_indices:
                        with open(os.path.join(path, str(valImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)  
                    else:
                        with open(os.path.join(path, str(trainImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)      

                else: 
                    if 100*(len(patientStructuresDict) - p) / len(patientStructuresDict) > 10:  #separate 90% of data into training set, other into validation
                        with open(os.path.join(path, str(trainImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)         
                        # with open(os.path.join(path, str(trainContourBoolPath)+ "_" + sliceText + ".txt"), "wb") as fp:
                        #     pickle.dump(contourOnPlane[zSlice], fp)          
                    else:
                        with open(os.path.join(path, str(valImagePath + "_" + sliceText + ".txt")), "wb") as fp:
                            pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)         
                        # with open(os.path.join(path, str(valContourBoolPath + "_" + sliceText + ".txt")), "wb") as fp:
                        #     pickle.dump(contourOnPlane[zSlice], fp)     
            # else:
            #     with open(os.path.join(path, str(testImagePath + "_" + sliceText + ".txt")), "wb") as fp:
            #             pickle.dump([combinedData[:,zSlice,:,:], slice_ZVals[zSlice]], fp)         
            #     with open(os.path.join(path, str(testContourBoolPath + "_" + sliceText + ".txt")), "wb") as fp:
            #  
            #           pickle.dump(contourOnPlane[zSlice], fp)  
    
    
    val_list = [] 
    for idx in val_indices:
        val_list.append(patientFolders[idx])
    return val_list      


def GetTrainingData_Tubarial(patients_folder, path, cross_val=False, fold=None):
    
    #first process left and right tubarial data as normally 
    val_list = GetTrainingData(patients_folder, "Left Tubarial", path, cross_val=cross_val, fold=fold)
    val_list = GetTrainingData(patients_folder, "Right Tubarial", path, cross_val=cross_val, fold=fold)

    print("Sorting Tubarial Data")
    trainImagePath = os.path.join(path, "Processed_Data", "Tubarial") 
    valImagePath = os.path.join(path, "Processed_Data", "Tubarial_Val")  

    trainImagePath_right = os.path.join(path, "Processed_Data", "Right Tubarial") 
    valImagePath_right = os.path.join(path, "Processed_Data", "Right Tubarial_Val")  

    trainImagePath_left = os.path.join(path, "Processed_Data", "Left Tubarial") 
    valImagePath_left = os.path.join(path, "Processed_Data", "Left Tubarial_Val")  


    #first delete currently processed files
    for file in os.listdir(valImagePath):
        os.remove(os.path.join(valImagePath, file))
    for file in os.listdir(trainImagePath):
        os.remove(os.path.join(trainImagePath, file))    

    #now copy the right tubarial data directly into tubarial folder
    for file in os.listdir(trainImagePath_right):
        file_path = os.path.join(trainImagePath_right, file)
        file_destination = os.path.join(trainImagePath, file)
        shutil.copy(file_path, file_destination)

    for file in os.listdir(valImagePath_right):
        file_path = os.path.join(valImagePath_right, file)
        file_destination = os.path.join(valImagePath, file)
        shutil.copy(file_path, file_destination)   

    #left tubarial data needs to be flipped horizontally     
    for file in os.listdir(trainImagePath_left):
        file_path = os.path.join(trainImagePath_left, file)
        file_destination = os.path.join(trainImagePath, str("l_" + file))
        data = pickle.load(open(file_path, "rb"))

        # print("before")
        # if np.amax(data[0][1,:,:]) > 0:
        #     fig, axs = plt.subplots(2)
        #     axs[0].imshow(data[0][0,:,:])
        #     axs[1].imshow(data[0][1,:,:])
        #     plt.show()

        data[0] = np.flip(data[0], 2)    #flip the order of the columns 

        # print("After")
        # if np.amax(data[0][1,:,:]) > 0:
            # fig, axs = plt.subplots(2)
            # axs[0].imshow(data[0][0,:,:])
            # axs[1].imshow(data[0][1,:,:])
            # plt.show()

        with open(file_destination, "wb") as fp:
            pickle.dump(data, fp)

    for file in os.listdir(valImagePath_left):
        file_path = os.path.join(valImagePath_left, file)
        file_destination = os.path.join(valImagePath, str("l_" + file))
        data = pickle.load(open(file_path, "rb"))
        data[0] = np.flip(data[0], 2)    #flip the order of the columns 
        
        with open(file_destination, "wb") as fp:
            pickle.dump(data, fp)        

    return val_list








def GetPredictionCTs(patientName, path):
    """Processes CT images for a given patient from a dicom file and saves them
       for the purpose of predicting. 
     
    Args:
        patientName (str): the name of the patient folder to process data for
        path (str): the path to the directory containing organogenesis folders 
            (Models, Processed_Data, etc.)

    """

    #This prepares data to be used for predicting and not training, so it is not necessary to supply contours corresponding to images
    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   

    patientPath = "Patient_Files/" + patientName
    #get the list of patient folders
    patientPath = os.path.join(path, patientPath)
    patientFolder = sorted(os.listdir(patientPath))

    #Can add new structures to this existing structure set if need be        
    patient_CTs = []

    for fileName in patientFolder:
        #get the modality: 
        fileNameAbsolute = os.path.join(os.getcwd(),"Patient_Files", patientName, fileName)
        try:
            patientData = pydicom.dcmread(fileNameAbsolute)
        except: continue    
        modality = patientData[0x0008,0x0060].value 
        if "CT" == modality: 
            patient_CTs.append(fileNameAbsolute)
    iop = dcmread(patient_CTs[0]).get("ImageOrientationPatient")
    ipp = dcmread(patient_CTs[0]).get("ImagePositionPatient")
    #also need the pixel spacing
    pixelSpacing = dcmread(patient_CTs[0]).get("PixelSpacing")
    sliceThickness = dcmread(patient_CTs[0]).get("SliceThickness")

    #Now I want numpy arrays of all these CT images. I save these in a list, where each element is itself a list, with index 0: CT image, index 1: z-value for that slice
    CTs = []
    ssFactor = 1 #at the moment don't need to upsample
    for item in pixelSpacing:
        item *= ssFactor
    for CTFile in patient_CTs:
        ctData = dcmread(CTFile)       
        rescaleSlope = ctData[0x0028, 0x1053].value
        rescaleIntercept = ctData[0x0028, 0x1052].value
        array = np.array(ctData.pixel_array)
        array = array * rescaleSlope + rescaleIntercept
        resizedArray = ImageUpsizer(array , ssFactor)
        #Now normalize the image so its values are between 0 and 1
        #Anything greater than 2500 Hounsfield units is likely artifact, so cap here.
        resizedArray = np.clip(resizedArray, -1000, 2500)
        resizedArray = NormalizeImage(resizedArray)
        #images are now normalized in the dataset, this allows for data augmentation to be performed 
        #resizedArray = NormalizeImage(resizedArray)            

        CTs.append([resizedArray, dcmread(CTFile).data_element("ImagePositionPatient"), pixelSpacing, sliceThickness])
    CTs.sort(key=lambda x:x[1][2]) #not necessarily in order, so sort according to z-slice.

    with open(os.path.join(path, str("Predictions_Patients/" + patientName  + "_Processed.txt")), "wb") as fp:
        pickle.dump(CTs, fp)  
    return CTs 

def GetDICOMContours(patientName, organ, path):
    """Gets the contours of a given organ from a given patient's 
       dicom file. 
     
    Args:
        patientName (str): the name of the patient folder containing 
            the dicom files (CT images) to get the contours of
        organ (str) the organ to get contours of
        path (str): the path to the directory containing 
            organogenesis folders (Models, Processed_Data, etc.)
        scalePixels (bool) True if the contours are to be scaled by their pixel spacing values (such as for when computing volumes)   

    Returns: 
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points (x, y, and z values) 
            at a specific z value

    """

    if path == None: #if no path supplied, assume that data folders are set up as default in the working directory. 
        path = pathlib.Path(__file__).parent.absolute()   

    patientPath = "Patient_Files/" + patientName
    #get the list of patient folders
    patientPath = os.path.join(path, patientPath)
    patientFolder = sorted(glob.glob(os.path.join(patientPath, "*")))

    structFiles = []    
    for fileName in patientFolder:
        if "ORG_STRUCT" in fileName:
            continue
        #get the modality: 
        try:
            patientData = pydicom.dcmread(fileName)
            modality = patientData[0x0008,0x0060].value 
            if "STRUCT" in modality:
                structFiles.append(fileName)  
        except:
             pass    
        
    #structsMeta = dcmread(structFile).data_element("ROIContourSequence")
    _, roiNumber, fileNum = FindStructure(structFiles, organ)   
    
    structsMeta = dcmread(structFiles[fileNum]).data_element("ROIContourSequence")
    #collect array of contour points for test plotting
    numContourPoints = 0 
    contours = []
    for contourInfo in structsMeta:
        if contourInfo.get("ReferencedROINumber") == roiNumber: #get the correct contour for the given organ
            for contoursequence in contourInfo.ContourSequence: 
                contours.append(contoursequence.ContourData)
                #But this is easier to work with if we convert from a 1d to a 2d list for contours ( [ [x1,y1,z1], [x2,y2,z2] ... ] )
                tempContour = []
                i = 0
                while i < len(contours[-1]):
  
                    x = float(contours[-1][i])
                    y = float(contours[-1][i + 1])
                    z = float(contours[-1][i + 2])
                    tempContour.append([x, y, z ])
                    i += 3
                    if (numContourPoints > 0):
                        pointListy = np.vstack((pointListy, np.array([x,y,z])))
                    else:
                        pointListy = np.array([x,y,z])   
                    numContourPoints+=1       
                contours[-1] = tempContour            
    return contours


                
def FindStructure(structureList, organ, invalidStructures = []):
    """Finds the matching structure to a given organ in a patient's
       dicom file metadata. 
     
    Args:
        structureList (List): a list of paths to RTSTRUCT files in the patient folder
        organ (str): the organ to find the matching structure for
        invaidStructures (list): a list of structures that the matching 
            structure cannot be, defaults to an empty list

    Returns: 
        str, int: the matching structure's name in the metadata, the 
            matching structure's ROI number in the metadata. Returns "", 
            1111 if no matching structure is found in the metadata
        
    """

    #Here we take the string for the desired structure (organ) and find the matching structure for each patient. 
    #The algorithm is to first make sure that the organ has a substring matching the ROI with at least 3 characters,
    #then out of the remaining possiblities, find top 3 closest fits with damerau levenshtein algorithm, then check to make sure that they are allowed to match according to rules defined in AllowedToMatch(). There should then ideally
    # be only one possible match, but if there are two, take the first in the list.   

    #Get a list of all structures in structure set: 
    unfilteredStructures = []

    for fileNum, file in enumerate(structureList):
        
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if element.get("ROIName").lower() not in invalidStructures:
                #first check if need to do special matching for limbus name:
                if element.get("ROIName").lower() == "sm_l" and organ  == "Left Submandibular":
                    roiNumber = element.get("ROINumber")
                    return element.get("ROIName").lower(), roiNumber, fileNum
                if element.get("ROIName").lower() == "sm_r" and organ  == "Right Submandibular":
                    roiNumber = element.get("ROINumber")
                    return element.get("ROIName").lower(), roiNumber, fileNum    
                unfilteredStructures.append(element.get("ROIName").lower())        
    #Now find which is the best fit.
    #First filter out any structures without at least a 3 character substring in common
    structures = []
    for structure in unfilteredStructures:
        valid = True
        if LongestSubstring(structure, organ) < 3:
            valid = False
        if not AllowedToMatch(organ, structure): 
            valid = False
        #Add to structures if valid
        if valid:
            structures.append(structure)
    #Now test string closeness to find
    closestStrings = [["",100],["",100],["",100]] #has to be in the top 3 closest strings to check next conditions
    for structure in structures:
        closeness = StringDistance(structure, organ)
        closestStrings.sort(key=itemgetter(1)) #Sort by the closeness value, and not the structure names
        for value in range(len(closestStrings)):
            if closeness < closestStrings[value][1]: #If closer than a value already in the top 3
                closestStrings[value] = [structure, closeness]
                break
    
    if len(closestStrings) == 0:
        return "", 1111, 0    
    #Now return the organ that is remaining and has closest string
    fileNum = -1
    for file in structureList:
        fileNum = fileNum + 1
        roiSequence = pydicom.dcmread(file).data_element("StructureSetROISequence")
        for element in roiSequence:
            if element.get("ROIName").lower() == closestStrings[0][0]:
                roiNumber = element.get("ROINumber")
    try:
        return closestStrings[0][0], roiNumber, fileNum
    except:
        return "", 1111, 0 #error code for unfound match.                                                                                                          

def AllowedToMatch(s1, s2):
    """Determines whether or not s1 and s2 are allowed to match 
       based on if they both contain the correct substrings.
     
    Args:
        s1 (str): first string to determine match
        s2 (str): second string to determine match

    Returns: 
        allowed (bool): True if the strings are allowed to match, 
            false otherwise
        
    """

    s1 = s1.lower()
    s2 = s2.lower()
    allowed = True
    keywords = []
    #You can't have only one organ with one of these keywords...
    keywords.append("prv")
    keywords.append("tub")
    keywords.append("brain")
    keywords.append("ptv")
    keywords.append("stem")
    keywords.append("node")
    keywords.append("cord")
    keywords.append("chi")
    keywords.append("opt")
    keywords.append("oral")
    keywords.append("nerv")
    keywords.append("par")
    keywords.append("globe")
    keywords.append("lip")
    keywords.append("cav")
    keywords.append("sub")
    keywords.append("test")
    keywords.append("fact")
    keywords.append("lacrim")
    keywords.append("constrict")
    keywords.append("esoph")
    keywords.append("couch")
    keywords.append("gtv")
    keywords.append("ctv")
    keywords.append("avoid")
    keywords.append("ORG_")


    
    #keywords can't be in only one of two string names: 
    for keyword in keywords:
        num = 0
        if keyword in s1:
            num += 1
        if keyword in s2:
            num += 1
        if num == 1:
            allowed = False        

    #Cant have left and no l in other, or right and no r
    if "left" in s1:
        if "l" not in s2:
            allowed = False      
    if "left" in s2:
        if "l" not in s1:
            allowed = False    
    #its tricky matching up left and right organs sometimes with all the conventions used... this makes sure that both are left or both are right
    if ("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("l_" == s1[0:2]) or ("_l" == s1[-2:]):
        if not (("lpar" in s2) or ("lsub" in s2) or ("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2) or ("l_" == s2[0:2])or ("lt " == s2[0:3]) or ("_l" == s2[-2:])):   
            allowed = False  
    if (("_l_" in s2) or (" l " in s2) or  (" l-" in s2) or ("-l-" in s2) or (" l_" in s2) or ("_l " in s2) or ("-l " in s2) or ("left" in s2) or ("l " == s2[0:2])or ("_lt_" in s2) or (" lt " in s2) or  (" lt-" in s2) or ("-lt-" in s2) or (" lt_" in s2) or ("_lt " in s2) or ("-lt " in s2) or ("l_" == s2[0:2])or ("lt " == s2[0:3]) or ("_l" == s2[-2:])):  
        if not (("lpar" in s1) or ("lsub" in s1) or ("_l_" in s1) or (" l " in s1) or  (" l-" in s1) or ("-l-" in s1) or (" l_" in s1) or ("_l " in s1) or ("-l " in s1) or ("left" in s1) or ("l " == s1[0:2]) or ("_lt_" in s1) or (" lt " in s1) or  (" lt-" in s1) or ("-lt-" in s1) or (" lt_" in s1) or ("l_" == s1[0:2]) or ("_lt " in s1) or ("-lt " in s1) or ("lt " == s1[0:3]) or ("_l" == s1[-2:])):
            allowed = False        
    
    if ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("-rt " in s1)or ("right" in s1) or ("r_" == s1[0:2])or ("_r" == s1[-2:]):
        if not (("rpar" in s2) or ("rsub" in s2) or ("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("r_" == s2[0:2]) or ("-rt" in s2) or ("_r" == s2[-2:])):   
            allowed = False
    if (("_r_" in s2) or (" r " in s2) or  (" r-" in s2) or ("-r-" in s2) or (" r_" in s2) or ("_r " in s2) or ("-r " in s2) or ("right" in s2) or ("r " == s2[0:2]) or ("_rt_" in s2) or (" rt " in s2) or  (" rt-" in s2) or ("-rt-" in s2) or (" rt_" in s2) or ("_rt " in s2) or ("-rt" in s2) or ("r_" == s2[0:2]) or ("_r" == s2[-2:])): 
        if not (("rpar" in s1) or ("rsub" in s1) or ("_r_" in s1) or (" r " in s1) or  (" r-" in s1) or ("-r-" in s1) or (" r_" in s1) or ("_r " in s1) or ("-r " in s1) or ("right" in s1) or ("r " == s1[0:2])or ("_rt_" in s1) or (" rt " in s1) or  (" rt-" in s1) or ("-rt-" in s1) or (" rt_" in s1) or ("_rt " in s1) or ("r_" == s1[0:2]) or ("-rt " in s1) or ("_r" == s1[-2:])):
            allowed = False
    return allowed


def StringDistance(s1, s2):
    """returns the Damerau-Levenshtein distance between two strings

    Args:
        s1 (string): string one which is to be compared with string 2.
        s2 (string): string two which is to be compared with string 1.

    Returns:
        (int): the Damerau Levenshtein distance between s1 and s2, which indicates how different the two strings are in terms of the amount of deletion, insertion, substitution, and transposition operations required to equate the two.

    """
    return damerauLevenshtein(s1,s2,similarity=False)

def LongestSubstring(s1,s2):
    """Finds the length of the longest substring that is in 
       both s1 and s2.
     
    Args:
        s1 (str): the first string to find the longest substring in
        s2 (str): the second string to find the longest substring in

    Returns: 
        longest (int): the length of the longest substring that is in 
            s1 and s2
        
    """

    m = len(s1)
    n = len(s2)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(s1[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(s1[i-c+1:i+1])
    return longest   



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

def ImageUpsizer(array, factor):
    """Supersizes an array by the factor given.   

    Args:
        array (2D numpy array): an array to be supersized
        factor (int): the amount to supersize the array by 

    Returns:
        newArray (2D numpy array): the supersized array

    """
    if factor == 1: 
        return array

    #Take an array and supersize it by the factor given
    xLen, yLen = array.shape
    newArray = np.zeros((factor * xLen, factor * yLen))
    #first get the original values in to the grid: 
    for i in range(xLen):
        for j in range(yLen):
            newArray[i * factor, j * factor] = array[i,j]
    #sample along first dim
    for j in range(yLen):
        for i in range(xLen - 1):
            insert = 1 
            while insert <= factor - 1:
                newArray[i * factor + insert, j * factor] = newArray[i * factor, j * factor] + (insert / factor) * (newArray[(i+1) * factor, j * factor]- newArray[i * factor, j * factor])
                insert += 1
    #sample along second dim
    for i in range(xLen * factor):
        for j in range(yLen - 1):
            insert = 1 
            while insert <= factor - 1:
                newArray[i, j * factor + insert] = newArray[i, j * factor] + (insert / factor) * (newArray[i, (j+1) * factor]- newArray[i, j * factor])
                insert += 1
    return newArray

def DICOM_to_Image_Coordinates(IPP, pixelSpacing, contours):
    """Converts the coordinates in contours from real world (dicom)
       coordinates to pixel coordinates. 

    Args:
        IPP (list): image position patient, the real world (dicom) 
            coordinate of the top left hand corner of the CT image
            [x,y,z]
        pixelSpacing (list): the real world (dicom) distance between 
            the centre of each pixel [x distance, y distance] in mm 
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value 
            [x,y,z] in real world (dicom) coordinates

    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value 
            [x,y,z] in  pixel coordinates

    """

    #Need to go throug each contour point and convert x and y values to integer values indicating image indices
    for contour in contours:
        for point in contour:
            point[0] = round((point[0] - IPP[0]) / pixelSpacing[0]) 
            point[1] = round((point[1] - IPP[1]) / pixelSpacing[1]) 
    return contours        

def PixelToContourCoordinates(contours, ipp, zValues, pixelSpacing, sliceThickness):
    """Converts the coordinates in contours from pixel coordinates
       to real world (dicom) coordinates. 

    Args:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value 
            [x,y] in pixel coordinates
        ipp (list): image position patient, the real world (dicom) 
            coordinate of the top left hand corner of the CT image
            [x,y,z]
        zValues (list): a list of the z values for the slices in contours
        pixelSpacing (list): the real world (dicom) distance between 
            the centre of each pixel [x distance, y distance] in mm 
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value 
            [x,y,z] in real world (dicom) coordinates
        sliceThickness (float): the distance between each CT slice in mm

    Returns:
        contours (list): a list of lists. Each item is a list 
            of the predicted contour points at a specific z value 
            [x,y,z] in real world (dicom) coordinates

    """

    newContours = []
    for layer in range(len(contours)): 
        newContours.append([])
        if len(contours[layer]) > 0:
            for point in contours[layer]:
                x  = round((point[0] * pixelSpacing[0]) + ipp[0],2)
                y  = round((point[1] * pixelSpacing[1]) + ipp[1],2)
                z = round(zValues[layer],2)
                newContours[-1].append([x,y,z])
    return newContours        

if __name__ == "__main__":
    print("Main Method of DicomParsing.py")
    patientsPath = 'Patient_Files/'
    GetTrainingData(patientsPath, "Spinal Cord")

    