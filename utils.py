# Function to parse the input files
import numpy as np
import scipy.special

def parseFile( fileName , fileType ):
    if( fileType != "data" and fileType != "weights" ):
        return
    try:
        fileObj = open( fileName , 'r')
    except IOError:
        return
    firstLine = np.array(fileObj.readline().strip('\n').split(' '))
    fileParams = firstLine.astype(np.int)
    if( fileType == "data" ):
        features = np.zeros( (fileParams[0], fileParams[1]) )
        targets = np.zeros( (fileParams[0], fileParams[2]) )
        try:
            for lineNum in range(0, fileParams[0]):
                thisLine = fileObj.readline().strip('\n').split(' ')
                features[lineNum] = thisLine[0:-1]
                targets[lineNum] = thisLine[-1]
        except ValueError:
            return
        return fileParams, features, targets
    
    # else fileType == weights
    weights_hidden = np.zeros( (fileParams[1],fileParams[0]+1) )
    weights_out = np.zeros( (fileParams[2],fileParams[1]+1) )
    try:
        for lineNum in range(0,fileParams[1]):
            weights_hidden[lineNum] = fileObj.readline().strip('\n').split(' ')
        for lineNum in range(0, fileParams[2]):
            weights_out[lineNum] = fileObj.readline().strip('\n').split(' ')
    except ValueError:
        return
    return fileParams, weights_hidden, weights_out

def sigmoid( x ):
    return scipy.special.expit( x )

def sigmoidPrime( x ):
    return sigmoid(x) * (1 - sigmoid(x))

