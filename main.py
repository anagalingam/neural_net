import numpy as np
import utils

trainObj = None
initObj = None
resFile = None
num_epochs = 0
learning_rate = 0

while( trainObj is None ):
    trainFile = raw_input("Enter the name of the training data file: ")
    trainObj = utils.parseFile(trainFile, "data")
    if( trainObj is None ):
        print "Invalid data file!\n"

while( initObj is None ):
    initFile = raw_input("Enter the name of the initialization file: ")
    initObj = utils.parseFile(initFile, "weights")
    if( initObj is None ):
        print "Invalid weights file!\n"

while( resFile is None ):
    resFileInput = raw_input("Enter the name of the trained weights output file: ")
    if( os.access(resFileInput, os.F_OK||os.W_OK) ):
        resFile = resFileInput 
    else:
        print "Invalid output file path!"

while( num_epochs < 1 ):
    try:
        num_epochs = int(raw_input("Enter the number of epochs for training: "))
    except ValueError:
        pass
    if( num_epochs < 1 ):
        print "Invalid number of epochs! Must be a positive integer."

while( learning_rate <= 0 ):
    try:
        learning_rate = float(raw_input("Enter the learning rate for training: "))
    except ValueError:
        pass
    if( learning_rate <= 0 ):
        print "Invalid learning rate! Must be a positive float (decimal notation)"


