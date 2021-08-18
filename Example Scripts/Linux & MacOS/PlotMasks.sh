#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
./Menu.py -o brainstem \
-f PlotMasks \
--modelType MultiResUNet \
--thres 0.2 \
