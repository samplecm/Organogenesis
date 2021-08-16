#!/usr/bin/env bash 
chmod +x Menu.py
./Menu.py -o cord \
-f Train \
--lr 0.001 \
--epochs 5 \
--modelType MultiResUNet \
--dataPath /media/calebsample/Data/patients \
 

