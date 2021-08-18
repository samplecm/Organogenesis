#!/usr/bin/env bash
cd ../.. 
chmod +x Menu.py
./Menu.py -o cord \
-f Train \
--lr 0.001 \
--epochs 10 \
--modelType UNet \
--processData
