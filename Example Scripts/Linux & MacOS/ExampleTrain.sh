#!/usr/bin/env bash
cd ../.. 
chmod +x Menu.py
<<<<<<< HEAD
./Menu.py -o left-parotid \
-f Train \
--lr 0.0001 \
--epochs 20 \
--modelType MultiResUNet \
--dataAugmentation \
#--loadModel
#--processData \
#--dataPath /media/calebsample/'TOSHIBA EXT'/Tubarial_Patients
=======
./Menu.py -o cord \
-f Train \
--lr 0.001 \
--epochs 10 \
--modelType UNet \
--processData
>>>>>>> 693bb8e6e02fee4e633aa734ed55dbb4af97d952
