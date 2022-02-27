@ECHO OFF
cd ../..
python Menu.py -o brainstem ^
-f PlotMasks ^
--modelType MultiResUNet ^
--thres 0.2
@pause

