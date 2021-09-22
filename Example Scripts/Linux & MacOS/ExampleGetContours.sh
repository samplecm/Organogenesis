#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
./Menu.py -o left-tubarial \
 -f GetContours \
 --modelType UNet \
 --predictionPatientName 1 \
