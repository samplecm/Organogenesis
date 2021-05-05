#Create all the necessary folders for the program to work correctly
pip install numpy
pip install Shapely
pip install pydicom
pip install open3d
pip install opencv-python
pip install pickle
pip install matplotlib
pip install pytorch
pip install SSIM-PIL
if [ ! -d "Loss History" ]; then
  mkdir "Loss History"
fi
if [ ! -d "Models" ]; then
  mkdir "Models"
fi
if [ ! -d "Patient_Files" ]; then
  mkdir "Patient_Files"
fi
if [ ! -d "Predictions_Patients" ]; then
  mkdir "Predictions_Patients"
fi
cd "Predictions_Patients"
if [ ! -d "Body" ]; then
  mkdir "Body"
fi
if [ ! -d "Spinal Cord" ]; then
  mkdir "Spinal Cord"
fi
if [ ! -d "Oral Cavity" ]; then
  mkdir "Oral Cavity"
fi
if [ ! -d "Left Parotid" ]; then
  mkdir "Left Parotid"
fi
if [ ! -d "Right Parotid" ]; then
  mkdir "Right Parotid"
fi



cd ../Processed_Data
if [ ! -d "Body" ]; then
  mkdir "Body"
fi
if [ ! -d "Spinal Cord" ]; then
  mkdir "Spinal Cord"
fi
if [ ! -d "Oral Cavity" ]; then
  mkdir "Oral Cavity"
fi
if [ ! -d "Left Parotid" ]; then
  mkdir "Left Parotid"
fi
if [ ! -d "Right Parotid" ]; then
  mkdir "Right Parotid"
fi

if [ ! -d "Body_Val" ]; then
  mkdir "Body_Val"
fi
if [ ! -d "Spinal Cord_Val" ]; then
  mkdir "Spinal Cord_Val"
fi
if [ ! -d "Oral Cavity_Val" ]; then
  mkdir "Oral Cavity_Val"
fi
if [ ! -d "Left Parotid_Val" ]; then
  mkdir "Left Parotid_Val"
fi
if [ ! -d "Right Parotid_Val" ]; then
  mkdir "Right Parotid_Val"
fi

if [ ! -d "Body bools" ]; then
  mkdir "Body bools"
fi
if [ ! -d "Spinal Cord bools" ]; then
  mkdir "Spinal Cord bools"
fi
if [ ! -d "Oral Cavity bools" ]; then
  mkdir "Oral Cavity bools"
fi
if [ ! -d "Left Parotid bools" ]; then
  mkdir "Left Parotid bools"
fi
if [ ! -d "Right Parotid bools" ]; then
  mkdir "Right Parotid bools"
fi

if [ ! -d "Body_Test" ]; then
  mkdir "Body_Test"
fi
if [ ! -d "Spinal Cord_Test" ]; then
  mkdir "Spinal Cord_Test"
fi
if [ ! -d "Oral Cavity_Test" ]; then
  mkdir "Oral Cavity_Test"
fi
if [ ! -d "Left Parotid_Test" ]; then
  mkdir "Left Parotid_Test"
fi
if [ ! -d "Right Parotid_Test" ]; then
  mkdir "Right Parotid_Test"
fi


cd ../
if [ ! -d "SavedImages" ]; then
  mkdir "SavedImages"
fi




# #already have patients 1-4 in the file, I just want to rename the other ones starting at 5.
# i=5
# for file in $LOCATION/*
#     do
#         #check if name starts with SGF
#         fileName="${file:0:3}"
#         echo "${fileName}"
#         if [$fileName -eq "SGB"]
#             then
#                 mv "$file" "$P$i"
#                 i=i + 1
#         fi
#     done 