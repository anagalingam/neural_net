# Function to parse the input files
import numpy as np

def parseFile( fileName , fileType ):
    if( fileType != "data" and fileType != "weights" ):
        return -1
    fileObj = open( fileName , 'r')
    firstLine = np.array(fileObj.readline().strip('\n').split(' '))
    fileParams = firstLine.astype(np.int)
    if( fileType == "data" ):
        features = np.zeros( (fileParams[0], fileParams[1]) )
        targets = np.zeros( (fileParams[0], fileParams[2]) )
        for lineNum in range(0, fileParams[0]):
            thisLine = fileObj.readline().strip('\n').split(' ')
            features[lineNum] = thisLine[0:-1]
            targets[lineNum] = thisLine[-1]
        return fileParams, features, targets
    
    # else fileType == weights
    weights_hidden = np.zeros( (fileParams[1],fileParams[0]+1) )
    weights_out = np.zeros( (fileParams[2],fileParams[1]+1) )
    for lineNum in range(0,fileParams[1]):
        weights_hidden[lineNum] = fileObj.readline().strip('\n').split(' ')
    for lineNum in range(0, fileParams[2]):
        weights_out[lineNum] = fileObj.readline().strip('\n').split(' ')
    return fileParams, weights_hidden, weights_out


result = parseFile( "docs/wdbc/data/init.txt", "weights")
print "The file params are: " + str(result[0])
print "The features are: " + str(result[1])
print "The targets are: " + str(result[2])
print "The type of features is: " + str(type(result[0]))
