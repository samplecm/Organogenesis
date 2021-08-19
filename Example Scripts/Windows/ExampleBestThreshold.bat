@ECHO OFF
cd ../..
python Menu.py -o cord ^
-f BestThreshold ^
--modelType MultiResUNet 
@pause

