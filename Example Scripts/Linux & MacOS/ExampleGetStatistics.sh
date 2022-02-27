#!/usr/bin/env bash 
cd ../..
chmod +x Menu.py
<<<<<<< HEAD
./Menu.py -o right-tubarial \
--modelType UNet\
 -f GetEvalData \
=======
./Menu.py -o cord \
--modelType UNet\
 -f GetEvalData \
 --thres 0.3
>>>>>>> 693bb8e6e02fee4e633aa734ed55dbb4af97d952
