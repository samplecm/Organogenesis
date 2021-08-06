import torch 
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

class UNet(nn.Module):
    """A class for the convolutional neural network UNet. 

    """

    def __init__(self):
      """Instantiation method for UNet. 

      """

      super(UNet, self).__init__()
      self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
      self.downConv1 = DoubleConv(1,64)
      self.downConv2 = DoubleConv(64,128)
      self.downConv3 = DoubleConv(128,256)
      self.downConv4 = DoubleConv(256, 512)
      self.downConv5 = DoubleConv(512,1024)
      self.upTrans1 = nn.ConvTranspose2d(in_channels=1024, out_channels = 512, kernel_size = 2, stride = 2)
      self.upConv1 = DoubleConv(1024, 512)
      self.upTrans2 = nn.ConvTranspose2d(in_channels=512, out_channels = 256, kernel_size = 2, stride = 2)
      self.upConv2 = DoubleConv(512, 256)
      self.upTrans3 = nn.ConvTranspose2d(in_channels=256, out_channels = 128, kernel_size = 2, stride = 2)
      self.upConv3 = DoubleConv(256, 128)
      self.upTrans4 = nn.ConvTranspose2d(in_channels=128, out_channels = 64, kernel_size = 2, stride = 2)
      self.upConv4 = DoubleConv(128, 64)
      self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, image):
        """Performs the U architecture (encoder and decoder) of UNet.

        Args:
            image (2D tensor): the input image
            
        Returns:
            x (2D tensor): the segmentation mask

        """

        #bs, c , h ,w
        #The encoder (first half of "U")
        x1 = self.downConv1(image) #
        x2 = self.maxPool_2x2(x1)
        x3 = self.downConv2(x2) #
        x4 = self.maxPool_2x2(x3)
        x5 = self.downConv3(x4)  #
        x6 = self.maxPool_2x2(x5)
        x7 = self.downConv4(x6)  #
        x8 = self.maxPool_2x2(x7)
        x9 = self.downConv5(x8)
        #Now the decoder part (second half of "U")CA

        #concatenate with x7, but first need to crop x7.
        x = self.upTrans1(x9)       
        y = Crop(x7, x)
        x = self.upConv1(torch.cat([x, y], 1))

        x = self.upTrans2(x)       
        y = Crop(x5, x)
        x = self.upConv2(torch.cat([x, y], 1))

        x = self.upTrans3(x)       
        y = Crop(x3, x)
        x = self.upConv3(torch.cat([x, y], 1))

        x = self.upTrans4(x)       
        y = Crop(x1, x)
        x = self.upConv4(torch.cat([x, y], 1))
        x = self.out(x)    
        return x

    def trainingStep(self, x, y):
        """Performs a training step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns: 
            loss (float): the binary cross entropy loss of the model 
                using training data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss

    def validationStep(self, x, y):
        """Performs a validation step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns:
            loss (float): the binary cross entropy loss of the model 
                using validation data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss

def DoubleConv(inC, outC):
    """Performs two successive 3x3 convolutions, each followed by 
        a batch normalization, and a ReLU.

    Args: 
        inC (int): input channels, the number of channels of the inputted tensor
        outC (int): output channels, the number of channels of the outputted tensor

    Returns:
        conv (2D tensor): the tensor that has been modified

    """

    conv = nn.Sequential(
        nn.Conv2d(inC, outC, kernel_size = 3, padding=1),
        nn.BatchNorm2d(outC),
        nn.ReLU(inplace=True),
        nn.Conv2d(outC, outC, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(outC),
        nn.ReLU(inplace=True)
    )    
    return conv

def Crop(tensor, targetTensor):
    """Crops tensor to the size of targetTensor. 

    Args: 
        tesnor (2D tensor): the tensor to be cropped
        targetTensor (int): the tensor of the desired size

    Returns:
        tensor: the cropped tensor 

    """

    targetSize= targetTensor.size()[2]
    tensorSize= tensor.size()[2]
    delta = tensorSize - targetSize
    delta = delta // 2
    return tensor[:, : , delta:tensorSize - delta, delta:tensorSize - delta]        


class MultiResBlock(nn.Module):
    """A class for the MultiResBlock which is a part of MultiResUNet. 

    """

    def __init__(self,inC, outC, alpha):
        """Instantiation method for the MultiResBlock class. 
        
        Args: 
            inC (int): input channels, the number of channels of the inputted tensor
            outC (int): output channels, the number of channels of the outputted tensor
            alpha (float): scalar cefficient which can be adjusted to change the 
                number of parameters in the model

        """

        super().__init__()
        self.inChannels = inC
        self.W = alpha * outC
        self.numFilters1 = int(self.W/6)
        self.numFilters2 = int(self.W/3)
        self.numFilters3 = int(self.W/2)
        self.finalFilters = int(self.numFilters1 + self.numFilters2 + self.numFilters3)
        self.skipConv = nn.Conv2d(inC, self.finalFilters, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(inC, self.numFilters1, kernel_size = 3, padding=1)
        self.conv2 = nn.Conv2d(self.numFilters1, self.numFilters2, kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(self.numFilters2, self.numFilters3, kernel_size = 3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(self.numFilters1)
        self.batchNorm2 = nn.BatchNorm2d(self.numFilters2)
        self.batchNorm3 = nn.BatchNorm2d(self.numFilters3)
        self.batchNormSkip = nn.BatchNorm2d(self.numFilters1+self.numFilters2+self.numFilters3)
        self.batchNormCat = nn.BatchNorm2d(self.numFilters1 + self.numFilters2 + self.numFilters3)
        self.batchNormFinal = nn.BatchNorm2d((self.numFilters1 + self.numFilters2 + self.numFilters3))

      
    def forward(self, image):
      """Performs the steps of the MultiResBlock. 

      Args: 
        image (2D tensor): the tensor to go through the MultiResBlock

      Returns: 
        final (2D tensor) the modified tensor 
      """

      conv1 = self.conv1(image)
      conv1 = self.batchNorm1(conv1)
      conv1 = nn.ReLU(inplace=True)(conv1)
      conv2 = self.conv2(conv1)
      conv2 = self.batchNorm2(conv2)
      conv2 = nn.ReLU(inplace=True)(conv2)
      conv3 = self.conv3(conv2)
      conv3 = self.batchNorm3(conv3)
      conv3 = nn.ReLU(inplace=True)(conv3)
      skip = self.skipConv(image)
      skip = self.batchNormSkip(skip)
      skip = nn.ReLU(inplace=True)(skip)
      cat = torch.cat([conv1, conv2, conv3], 1)
      cat = self.batchNormCat(cat) 
      final = cat + skip
      final = nn.ReLU(inplace=True)(final)
      final = self.batchNormFinal(final)
      return final

class MultiResUNet(nn.Module):
    """A class for the convolutional neural network MultiResUNet. 

    """

    def __init__(self):
        """Instantiation method for MultiResUNet. 

        """
        super().__init__()
        self.alpha = 1.67
        self.inSize2 = int(self.alpha*32/6) + int(self.alpha*32/3) + int(self.alpha*32/2)
        self.inSize3 = int(self.alpha*64/6) + int(self.alpha*64/3) + int(self.alpha*64/2)
        self.inSize4 = int(self.alpha*128/6) + int(self.alpha*128/3) + int(self.alpha*128/2)
        self.inSize5 = int(self.alpha*256/6) + int(self.alpha*256/3) + int(self.alpha*256/2)
        self.inSize6 = int(self.alpha*512/6) + int(self.alpha*512/3) + int(self.alpha*512/2)
        self.inSize7 = self.inSize5 * 2
        self.inSize8 = self.inSize4 * 2
        self.inSize9 = self.inSize3 * 2
        self.inSize10 = self.inSize2 * 2

        self.multiRes1 = MultiResBlock(1,32, self.alpha)
        self.multiRes2 = MultiResBlock(self.inSize2,64, self.alpha)
        self.multiRes3 = MultiResBlock(self.inSize3, 128, self.alpha)
        self.multiRes4 = MultiResBlock(self.inSize4, 256, self.alpha)
        self.multiRes5 = MultiResBlock(self.inSize5,512, self.alpha)
        self.multiRes6 = MultiResBlock(self.inSize7,256, self.alpha)
        self.multiRes7 = MultiResBlock(self.inSize8,128, self.alpha)
        self.multiRes8 = MultiResBlock(self.inSize9,64, self.alpha)
        self.multiRes9 = MultiResBlock(self.inSize10,32, self.alpha)

        self.maxPool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upTrans1 = nn.ConvTranspose2d(in_channels=self.inSize6, out_channels = self.inSize5, kernel_size = 2, stride = 2, padding=0)
        self.upTrans2 = nn.ConvTranspose2d(in_channels=self.inSize5, out_channels = self.inSize4, kernel_size = 2, stride = 2, padding=0)
        self.upTrans3 = nn.ConvTranspose2d(in_channels=self.inSize4, out_channels = self.inSize3, kernel_size = 2, stride = 2, padding=0)
        self.upTrans4 = nn.ConvTranspose2d(in_channels=self.inSize3, out_channels = self.inSize2, kernel_size = 2, stride = 2, padding=0)
        self.out = nn.Conv2d(in_channels=self.inSize2, out_channels=1, kernel_size=1, padding=0)

        self.resPath4_Conv3 = nn.Conv2d(self.inSize5, self.inSize5, kernel_size = 3, padding=1)
        self.resPath4_Conv1 = nn.Conv2d(self.inSize5, self.inSize5, kernel_size = 1, padding=0)
        self.resPath4_batchNorm = nn.BatchNorm2d(self.inSize5)
        
        self.resPath3_Conv3 = nn.Conv2d(self.inSize4, self.inSize4, kernel_size = 3, padding=1)
        self.resPath3_Conv1 = nn.Conv2d(self.inSize4, self.inSize4, kernel_size = 1, padding=0)
        self.resPath3_batchNorm = nn.BatchNorm2d(self.inSize4)

        self.resPath2_Conv3 = nn.Conv2d(self.inSize3, self.inSize3, kernel_size = 3, padding=1)
        self.resPath2_Conv1 = nn.Conv2d(self.inSize3, self.inSize3, kernel_size = 1, padding=0)
        self.resPath2_batchNorm = nn.BatchNorm2d(self.inSize3)

        self.resPath1_Conv3 = nn.Conv2d(self.inSize2, self.inSize2, kernel_size = 3, padding=1)
        self.resPath1_Conv1 = nn.Conv2d(self.inSize2, self.inSize2, kernel_size = 1, padding=0)
        self.resPath1_batchNorm = nn.BatchNorm2d(self.inSize2)
   

    def forward(self, image):
        """Performs the U architecture (encoder, decoder, and Res paths) of MultiResUNet.

        Args:
            image (2D tensor): the input image
            
        Returns:
            x (2D tensor): the segmentation mask

        """

        #bs, c , h ,w
        #The encoder (first half of "U")
        x1 = self.multiRes1(image) #
        x2 = self.maxPool_2x2(x1)
        x3 = self.multiRes2(x2) #
        x4 = self.maxPool_2x2(x3)
        x5 = self.multiRes3(x4)  #
        x6 = self.maxPool_2x2(x5)
        x7 = self.multiRes4(x6)  #
        x8 = self.maxPool_2x2(x7)
        x9 = self.multiRes5(x8)
        #Now the decoder part (second half of "U")CA
        del(x2)
        del(x4)
        del(x6)
        del(x8)
        #concatenate with x7, but first need to crop x7.
        x = self.upTrans1(x9)   

        #Res path 4:
        xRes = self.resPath4_Conv3(x7)    
        xSkip = self.resPath4_Conv1(x7)
        xRes = xRes + xSkip       
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath4_batchNorm(xRes)

        x = self.multiRes6(torch.cat([x, xRes], 1))

        x = self.upTrans2(x)  
        
        #Res path 3:         
        xRes = self.resPath3_Conv3(x5)    
        xSkip = self.resPath3_Conv1(x5)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath3_batchNorm(xRes)

        xRes = self.resPath3_Conv3(xRes)    
        xSkip = self.resPath3_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath3_batchNorm(xRes)
        

        x = self.multiRes7(torch.cat([x, xRes], 1))

        x = self.upTrans3(x)   
        #Res path 2:    
        xRes = self.resPath2_Conv3(x3)    
        xSkip = self.resPath2_Conv1(x3)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath2_batchNorm(xRes)
        

        xRes = self.resPath2_Conv3(xRes)    
        xSkip = self.resPath2_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath2_batchNorm(xRes)
        

        xRes = self.resPath2_Conv3(xRes)    
        xSkip = self.resPath2_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath2_batchNorm(xRes)
        

        x = self.multiRes8(torch.cat([x, xRes], 1))

        x = self.upTrans4(x)    
        #Res path 1:    
        xRes = self.resPath1_Conv3(x1)    
        xSkip = self.resPath1_Conv1(x1)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath1_batchNorm(xRes)
        

        xRes = self.resPath1_Conv3(xRes)    
        xSkip = self.resPath1_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath1_batchNorm(xRes)
        

        xRes = self.resPath1_Conv3(xRes)    
        xSkip = self.resPath1_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath1_batchNorm(xRes)
        

        xRes = self.resPath1_Conv3(xRes)    
        xSkip = self.resPath1_Conv1(xRes)
        xRes = xRes + xSkip
        xRes = nn.ReLU(inplace=True)(xRes)
        xRes = self.resPath1_batchNorm(xRes)
        
        del(xSkip)
        x = self.multiRes9(torch.cat([x, xRes], 1))
        x = self.out(x)    
        return x

    def trainingStep(self, x, y):
        """Performs a training step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns:
            loss (float): the binary cross entropy loss of the model 
                using training data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss

    def validationStep(self, x, y):
        """Performs a validation step. 

        Args: 
            x (2D tensor): the CT image 
            y (2D tensor): the ground truth mask

        Returns:
            loss (float): the binary cross entropy loss of the model 
                using validation data

        """

        out = self(x)
        loss = nn.BCEWithLogitsLoss()(out.float(), y.float())    #
        return loss



