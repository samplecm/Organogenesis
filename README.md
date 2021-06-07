# Organogenesis
The program can be run from the Terminal, providing certain arguments corresponding to the desired task.
The best way to interact with the program is with a shell script, such as "TrainScript.sh" 
instructions for running:
in the program directory, first give permissions to TrainScript.sh with the command: 
$>chmod 777 TrainScript.sh
Then run the program by simply typing:
$> ./TrainScript.sh

Before this, to install dependencies and set up directories in the Organogenesis directory, first run FolderSetup.sh.
This can be done in the terminal with 
$> chmod 777 FolderSetup.sh
$>./FolderSetup.sh

Put patient folders into Patient_Files in separate folders names "P#" where "#" is an integer. In this folder, have all the CTs, and the RTSTRUCT file.
#Right now the program starts with the Menu.py file. This lets you choose an organ and then decide if you want to train, test, run stats etc. 




The program can also be run by simply typing 
$>python Menu.py (followed by optional arguments)
or perhaps
$>python3 Menu.py

There are certain keywords that are provided when TrainScript starts up Menu.py, or when Menu.py is directly launched.

To get some quick information about the options, type

$>python Menu.py -h

The two main arguments that can be provided are 

1. Organ (what organ you're interested in doing stuff with)
2. Function (What you want to do)

These are set by adding 
1. -o nameofOrgan
2. -f nameofFunction
right after python Menu.py in the same line.


To get the list of options, use Menu.py -h
There are also other function arguments which can be supplied, and some of them have default values. 

All other arguments have two dashes (--) followed by the argument keyword without a space, and then the value. See TrainScript.sh as an example, and use Menu.py to view the other possible options.

One noteable keyword is --dataPath, which can be set to a directory containing patient folders, which themselves have all the DICOM files in the same directory. If this keyword is set, then data will all be written to folders in this dataPath, rather than
the folders in the Organogenesis directory.

If --dataPath is not provided, then it is expected that patient Files have been placed in the "Patient_Files" directory in the Organogenesis directory. 

