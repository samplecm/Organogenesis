#!/bin/bash
#Create all the necessary folders for the program to work correctly
if [ ! -d "Statistics" ]; then
  mkdir "Statistics"
fi
if [ ! -d "Evaluation Data" ]; then
  mkdir "Evaluation Data"
fi
if [ ! -d "Loss History" ]; then
  mkdir "Loss History"
fi
cd "Loss History"
if [ ! -d "Body" ]; then
  mkdir "Body"
fi
if [ ! -d "Brainstem" ]; then
  mkdir "Brainstem"
fi
if [ ! -d "Brain" ]; then
  mkdir "Brain"
fi
if [ ! -d "Brachial Plexus" ]; then
  mkdir "Brachial Plexus"
fi
if [ ! -d "Chiasm" ]; then
  mkdir "Chiasm"
fi
if [ ! -d "Esophagus" ]; then
  mkdir "Esophagus"
fi
if [ ! -d "Globes" ]; then
  mkdir "Globes"
fi
if [ ! -d "Larynx" ]; then
  mkdir "Larynx"
fi
if [ ! -d "Lens" ]; then
  mkdir "Lens"
fi
if [ ! -d "Lips" ]; then
  mkdir "Lips"
fi
if [ ! -d "Mandible" ]; then
  mkdir "Mandible"
fi
if [ ! -d "Optic Nerves" ]; then
  mkdir "Optic Nerves"
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
if [ ! -d "Right Submandibular" ]; then
  mkdir "Right Submandibular"
fi
if [ ! -d "Left Submandibular" ]; then
  mkdir "Left Submandibular"
fi
if [ ! -d "Right Tubarial" ]; then
  mkdir "Right Tubarial"
fi
if [ ! -d "Left Tubarial" ]; then
  mkdir "Left Tubarial"
fi
if [ ! -d "Tubarial" ]; then
  mkdir "Tubarial"
fi


cd ../

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
if [ ! -d "Brainstem" ]; then
  mkdir "Brainstem"
fi
if [ ! -d "Brain" ]; then
  mkdir "Brain"
fi
if [ ! -d "Brachial Plexus" ]; then
  mkdir "Brachial Plexus"
fi
if [ ! -d "Chiasm" ]; then
  mkdir "Chiasm"
fi
if [ ! -d "Esophagus" ]; then
  mkdir "Esophagus"
fi
if [ ! -d "Globes" ]; then
  mkdir "Globes"
fi
if [ ! -d "Larynx" ]; then
  mkdir "Larynx"
fi
if [ ! -d "Lens" ]; then
  mkdir "Lens"
fi
if [ ! -d "Lips" ]; then
  mkdir "Lips"
fi
if [ ! -d "Mandible" ]; then
  mkdir "Mandible"
fi
if [ ! -d "Optic Nerves" ]; then
  mkdir "Optic Nerves"
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
if [ ! -d "Right Submandibular" ]; then
  mkdir "Right Submandibular"
fi

if [ ! -d "Left Submandibular" ]; then
  mkdir "Left Submandibular"
fi
if [ ! -d "Right Tubarial" ]; then
  mkdir "Right Tubarial"
fi
if [ ! -d "Left Tubarial" ]; then
  mkdir "Left Tubarial"
fi
cd ..

if [ ! -d "Processed_Data" ]; then
  mkdir "Processed_Data"
fi
cd "Processed_Data"
if [ ! -d "Sorted Contour Lists" ]; then
  mkdir "Sorted Contour Lists"
fi
if [ ! -d "Area Stats" ]; then
  mkdir "Area Stats"
fi
if [ ! -d "Body" ]; then
  mkdir "Body"
fi
if [ ! -d "Brainstem" ]; then
  mkdir "Brainstem"
fi
if [ ! -d "Brainstem_Val" ]; then
  mkdir "Brainstem_Val"
fi
if [ ! -d "Brainstem_Test" ]; then
  mkdir "Brain Stem_Test"
fi
if [ ! -d "Brainstem bools" ]; then
  mkdir "Brainstem bools"
fi
if [ ! -d "Left Submandibular" ]; then
  mkdir "Left Submandibular"
fi
if [ ! -d "Left Submandibular_Val" ]; then
  mkdir "Left Submandibular_Val"
fi
if [ ! -d "Left Submandibular_Test" ]; then
  mkdir "Left Submandibular_Test"
fi
if [ ! -d "Left Submandibular bools" ]; then
  mkdir "Left Submandibular bools"
fi
if [ ! -d "Right Submandibular" ]; then
  mkdir "Right Submandibular"
fi
if [ ! -d "Right Submandibular_Val" ]; then
  mkdir "Right Submandibular_Val"
fi
if [ ! -d "Right Submandibular_Test" ]; then
  mkdir "Right Submandibular_Test"
fi
if [ ! -d "Right Submandibular bools" ]; then
  mkdir "Right Submandibular bools"
fi
if [ ! -d "Brain" ]; then
  mkdir "Brain"
fi
if [ ! -d "Brain_Test" ]; then
  mkdir "Brain_Test"
fi
if [ ! -d "Brain_Val" ]; then
  mkdir "Brain_Val"
fi
if [ ! -d "Brain bools" ]; then
  mkdir "Brain bools"
fi
if [ ! -d "Brachial Plexus_Test" ]; then
  mkdir "Brachial Plexus_Test"
fi
if [ ! -d "Brachial Plexus_Val" ]; then
  mkdir "Brachial Plexus_Val"
fi
if [ ! -d "Brachial Plexus bools" ]; then
  mkdir "Brachial Plexus bools"
fi
if [ ! -d "Brachial Plexus" ]; then
  mkdir "Brachial Plexus"
fi
if [ ! -d "Chiasm" ]; then
  mkdir "Chiasm"
fi
if [ ! -d "Chiasm_Test" ]; then
  mkdir "Chiasm_Test"
fi
if [ ! -d "Chiasm_Val" ]; then
  mkdir "Chiasm_Val"
fi
if [ ! -d "Chiasm bools" ]; then
  mkdir "Chiasm bools"
fi
if [ ! -d "Esophagus" ]; then
  mkdir "Esophagus"
fi
if [ ! -d "Esophagus bools" ]; then
  mkdir "Esophagus bools"
fi
if [ ! -d "Esophagus_Test" ]; then
  mkdir "Esophagus_Test"
fi
if [ ! -d "Esophagus_Val" ]; then
  mkdir "Esophagus_Val"
fi
if [ ! -d "Globes" ]; then
  mkdir "Globes"
fi
if [ ! -d "Globes_Test" ]; then
  mkdir "Globes_Test"
fi
if [ ! -d "Globes_Val" ]; then
  mkdir "Globes_Val"
fi
if [ ! -d "Globes bools" ]; then
  mkdir "Globes bools"
fi
if [ ! -d "Larynx" ]; then
  mkdir "Larynx"
fi
if [ ! -d "Larynx bools" ]; then
  mkdir "Larynx bools"
fi
if [ ! -d "Larynx_Test" ]; then
  mkdir "Larynx_Test"
fi
if [ ! -d "Larynx_Val" ]; then
  mkdir "Larynx_Val"
fi
if [ ! -d "Lens" ]; then
  mkdir "Lens"
fi
if [ ! -d "Lens_Val" ]; then
  mkdir "Lens_Val"
fi
if [ ! -d "Lens bools" ]; then
  mkdir "Lens bools"
fi
if [ ! -d "Lens_Test" ]; then
  mkdir "Lens_Test"
fi
if [ ! -d "Lips" ]; then
  mkdir "Lips"
fi
if [ ! -d "Lips_Val" ]; then
  mkdir "Lips_Val"
fi
if [ ! -d "Lips_Test" ]; then
  mkdir "Lips_Test"
fi
if [ ! -d "Lips bools" ]; then
  mkdir "Lips bools"
fi
if [ ! -d "Mandible" ]; then
  mkdir "Mandible"
fi
if [ ! -d "Mandible_Test" ]; then
  mkdir "Mandible_Test"
fi
if [ ! -d "Mandible_Val" ]; then
  mkdir "Mandible_Val"
fi
if [ ! -d "Mandible bools" ]; then
  mkdir "Mandible bools"
fi
if [ ! -d "Optic Nerves" ]; then
  mkdir "Optic Nerves"
fi
if [ ! -d "Optic Nerves_Val" ]; then
  mkdir "Optic Nerves_Val"
fi
if [ ! -d "Optic Nerves_Test" ]; then
  mkdir "Optic Nerves_Test"
fi
if [ ! -d "Optic Nerves bools" ]; then
  mkdir "Optic Nerves bools"
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
if [ ! -d "Right Tubarial" ]; then
  mkdir "Right Tubarial"
fi
if [ ! -d "Right Tubarial_Test" ]; then
  mkdir "Right Tubarial_Test"
fi
if [ ! -d "Right Tubarial_Val" ]; then
  mkdir "Right Tubarial_Val"
fi
if [ ! -d "Right Tubarial bools" ]; then
  mkdir "Right Tubarial bools"
fi
if [ ! -d "Left Tubarial" ]; then
  mkdir "Left Tubarial"
fi
if [ ! -d "Left Tubarial_Test" ]; then
  mkdir "Left Tubarial_Test"
fi
if [ ! -d "Left Tubarial_Val" ]; then
  mkdir "Left Tubarial_Val"
fi
if [ ! -d "Left Tubarial bools" ]; then
  mkdir "Left Tubarial bools"
fi
if [ ! -d "Tubarial" ]; then
  mkdir "Tubarial"
fi
if [ ! -d "Tubarial_Test" ]; then
  mkdir "Tubarial_Test"
fi
if [ ! -d "Tubarial_Val" ]; then
  mkdir "Tubarial_Val"
fi
if [ ! -d "Tubarial bools" ]; then
  mkdir "Tubarial bools"
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