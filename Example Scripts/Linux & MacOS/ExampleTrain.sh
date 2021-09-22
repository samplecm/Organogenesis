#!/usr/bin/env bash
cd ../.. 
chmod +x Menu.py
./Menu.py -o Left-Tubarial \
-f Train \
--lr 0.0001 \
--epochs 10 \
--modelType UNet \
--processData \
--dataPath /media/calebsample/'TOSHIBA EXT'/Tubarial_Patients
