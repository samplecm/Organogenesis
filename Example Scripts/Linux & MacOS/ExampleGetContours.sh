#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
<<<<<<< HEAD
./Menu.py -o Body \
 -f GetContours \
 --modelType UNet \
 --predictionPatientName Tubarial_Ecl \
=======
./Menu.py -o all \
 -f GetContours \
 --modelType MultiResUNet \
 --predictionPatientName patientName \
>>>>>>> 693bb8e6e02fee4e633aa734ed55dbb4af97d952
