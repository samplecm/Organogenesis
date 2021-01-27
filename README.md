# Organogenesis
First run the FolderSetup.sh script to make sure that the folders are all set up right, and install dependencies.
Put patient folders into Patient_Files in separate folders names "P#" where "#" is an integer. In this folder, have all the CTs, and the RTSTRUCT file.
#Right now the program starts with the Menu.py file. This lets you choose an organ and then decide if you want to train, test, run stats etc. 

Right now, if you train it will plot a point cloud of each patients structure first so you can decide if you want to use it for training or not. If you uncomment 200-222 in DicomParsing.py it will also plot the body contour (every second slice) with it for perspective.


