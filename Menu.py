#local dependencies:
import DicomParsing
import Train
import Test
import Predict
import PostProcessing

def main():
    print("Welcome to Organogenesis")
    print("------------------")

    #Keep a list of organs which can be contoured
    OARs = ["Body", "Spinal Cord", "Oral Cavity", "Left Parotid", "Right Parotid"] #Later will add an "all" option

    #Need to get user input. Make a string to easily ask for a number corresponding to an OAR.
    ChooseOAR_string = "Please enter the number for the organ you wish to contour / train a model for \n>>"
    for i in range(len(OARs)):
        ChooseOAR_string += str(i + 1) + ": " + OARs[i] + "\n"
    while True:
        try:
            chosenOAR = int(input(ChooseOAR_string)) - 1
            if chosenOAR < len(OARs):
                break
        except KeyboardInterrupt:
            quit()    
        except: pass     

    #Now determine if the goal is to train or to find contours
    chooseTask_string = "Please enter the number for the desired task\n"
    chooseTask_string += "1. Train a UNet model for predicting " + str(OARs[chosenOAR])
    chooseTask_string += "\n2. Predict " + str(OARs[chosenOAR]) + " contours using an existing model"
    chooseTask_string += "\n3. Determine threshold accuracies for predictions of the " + str(OARs[chosenOAR])
    chooseTask_string += "\n4. Determine F-score for validation set of the " + str(OARs[chosenOAR])
    chooseTask_string += "\n5. Plot predicted masks for the  " + str(OARs[chosenOAR])
    chooseTask_string += "\n6. Export model to ONNX"
    chooseTask_string += "\n7. predict using ONNX model \n>>"
    while True:
        try:
            task = int(input(chooseTask_string))
            if (task in range(0,8)):
                break
        except KeyboardInterrupt:
            quit()
        except: pass   

    if (task == 1):
        Train.Train(OARs[chosenOAR], 12, 1e-3, processData=False, loadModel=False)
        Test.Best_Threshold(OARs[chosenOAR],400)

        Test.TestPlot(OARs[chosenOAR], threshold=0.1)  
    elif task == 2:    
        Predict.GetContours(OARs[chosenOAR],"P86", threshold = 0.09, withReal=True, tryLoad=False) 
    elif task == 3:
        Test.Best_Threshold(OARs[chosenOAR],  testSize=500, onlyMasks=False,onlyBackground=False)
    elif task == 4:
        F_Score, recall, precision, accuracy = Test.FScore(OARs[chosenOAR], threshold=0.2)    
        print([F_Score, recall, precision, accuracy])
    elif task == 5:
        Test.TestPlot(OARs[chosenOAR], threshold=0.2) 
    elif task == 6:
        PostProcessing.Export_To_ONNX(OARs[chosenOAR])    
    elif task == 7:
        PostProcessing.Infer_From_ONNX(OARs[chosenOAR], 'P7')        
        

   

if __name__ == "__main__":
    main()