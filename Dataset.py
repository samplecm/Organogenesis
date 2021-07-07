import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

class CTDataset(Dataset):
    def __init__(self, dataFiles, root_dir, transform):
        self.dataFiles = dataFiles
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return(len(self.dataFiles))

    def __getitem__(self, index):
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
        #if its entirely ones or entirely zeros, can just return it as is. 
        if np.amin(image) == 1 and np.amax(image) == 1:
            return image
        elif np.amin(image) == 0 and np.amax(image) == 0:
            return image     
        ptp = np.ptp(image)
        amin = np.amin(image)
        return (image - amin) / ptp    