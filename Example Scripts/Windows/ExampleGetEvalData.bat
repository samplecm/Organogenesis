@ECHO OFF
cd ../..
python Menu.py -o cord ^
-f GetEvalData ^
--modelType UNet ^
--thres 0.3 
@pause

