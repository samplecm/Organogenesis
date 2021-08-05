import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

class CTDataset(Dataset):
    """A Dataset that stores CT images and their masks.

    """

    def __init__(self, dataFiles, root_dir, transform):
        """Instantiation method for CTDataset.

        Args:
            dataFiles (list): a list containing the names of the files with CT 
                data (images and masks)
            root_dir (str): the path to the driectory containing the dataFiles
            transform (Compose, None): the transform(s) to be used for data 
                augmentation

        """

        self.dataFiles = dataFiles
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        """Gets how many images there are in the dataset.

        """
        return(len(self.dataFiles))

    def __getitem__(self, index):
        """Gets the image and mask in the dataset at a given index.
           Performs data if it augmentation if being used.

        Args:
            index (int): the index of the image and mask to return

        Returns:
            image (3D tensor): a normalized tensor of the image 
                (num channels x height x width)
            mask (3D tensor): a tensor of the mask
                (num channels x height x width)
        """
        dataPath = os.path.join(self.root_dir, self.dataFiles[index])
        data = pickle.load(open(dataPath, 'rb'))
        image = data[0][0,:,:]
        mask = data[0][1,:,:]

        if self.transform: #performs data augmentation if we are using it 
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]

        xLen, yLen = image.shape

        image = self.NormalizeImage(image)

        #converts to tensors 
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        #reshapes the tensors to be num channels x height x width
        image = torch.reshape(image, (1, xLen, yLen)).float()
        mask = torch.reshape(mask, (1, xLen, yLen)).float()

        return(image, mask)

    def NormalizeImage(self, image):
        """Normalizes an image between 0 and 1.  

        Args:
            image (2D numpy array): the image to be normalized 

        Returns:
            2D numpy array: the normalized image

        """

        #if its entirely ones or entirely zeros, can just return it as is. 
        if np.amin(image) == 1 and np.amax(image) == 1:
            return image
        elif np.amin(image) == 0 and np.amax(image) == 0:
            return image     
        ptp = np.ptp(image)
        amin = np.amin(image)
        return (image - amin) / ptp    