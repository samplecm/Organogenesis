@ECHO OFF
cd ../..
python Menu.py -o all ^
-f GetContours ^
--modelType MultiResUNet ^
--predictionPatientName patientName 
@pause


