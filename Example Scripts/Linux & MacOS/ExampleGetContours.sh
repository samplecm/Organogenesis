#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
./Menu.py -o all \
 -f GetContours \
 --modelType MultiResUNet \
 --predictionPatientName patientName \
