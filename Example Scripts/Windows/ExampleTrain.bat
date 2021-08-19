@ECHO OFF
cd ../..
python Menu.py -o cord ^
-f Train ^
--lr 0.001 ^
--epochs 10 ^
--modelType UNet ^
--processData 
@pause


