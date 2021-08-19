#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
./Menu.py -o cord \
--modelType UNet\
 -f GetEvalData \
 --thres 0.3
