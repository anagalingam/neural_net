import numpy as np
import os
import utils
import pdb

debug = 0

#-------------------------------------------------------------------------------
## Input Parsing

initObj = None
testObj = None
resFile = None

while( initObj is None ):
    initFile = raw_input("Enter the name of the initialization file: ")
    initObj = utils.parseFile(initFile, "weights")
    if( initObj is None ):
        print "Invalid weights file!\n"

while( testObj is None ):
    testFile = raw_input("Enter the name of the training data file: ")
    testObj = utils.parseFile(testFile, "data")
    if( testObj is None ):
        print "Invalid data file!\n"

while( resFile is None ):
    resFileInput = raw_input("Enter the name of the trained weights output file: ")
#    if( os.access(resFileInput, os.W_OK) ):
    resFile = resFileInput 
#    else:
#        print "Invalid output file path!"

#------------------------------------------------------------------------------
## Initializing variables

in_layer = np.zeros( initObj[0][0]+1 )
hidden_layer = np.zeros( initObj[0][1]+1 )
out_layer = np.zeros( initObj[0][2] )

in2h_weights = initObj[1]
h2o_weights = initObj[2]

num_data = trainObj[0][0]
features = trainObj[1]
targets = trainObj[2]

contingency = np.zeros( (len(out_layer), 4) )   # Each class has array [A,B,C,D]


#-----------------------------------------------------------------------------
## Test data prediction

for data in range(0, num_data):
        # Forward Propagation of training data, in -> out layer
        in_layer[1:] = features[data]
        in_layer[0] = -1    # Setting the bias input for hidden layer
        
        hidden_accumulator = np.zeros( len(hidden_layer) )
        for hidden_node in range(1, len(hidden_layer)):
            hidden_accumulator[hidden_node] = np.dot(in2h_weights[hidden_node-1],in_layer)
            hidden_layer[hidden_node] = utils.sigmoid( hidden_accumulator[hidden_node] )
        
        hidden_layer[0] = -1    # Setting the bias input for output layer
        out_accumulator = np.zeros( len(out_layer) )
        for out_node in range(0, len(out_layer)):
            out_accumulator[out_node] = np.dot(h2o_weights[out_node],hidden_layer)
            out_layer[out_node] = utils.sigmoid( out_accumulator[out_node] )
            # Increment the appropriate contingency table entry
            if(out_layer[out_node] >= 0.5):
                if( targets[data] > 0.5 ):
                    contingency[out_node][0] += 1
                else:
                    contingency[out_node][1] += 1
            else:
                if( targets[data] > 0.5 ):
                    contingency[out_node][2] += 1
                else:
                    contingency[out_node][3] += 1

#----------------------------------------------------------------------------
## Write the trained weights output file

with open(resFile, 'w') as file:
    file.write(' '.join('%s' % ii for ii in initObj[0]))
    for hidden_node in range(0, initObj[0][1]):
        file.write('\n' + ' '.join('%.3f' % ii for ii in in2h_weights[hidden_node]))
    for out_node in range(0, initObj[0][2]):
        file.write('\n' + ' '.join('%.3f' % ii for ii in h2o_weights[out_node]))


