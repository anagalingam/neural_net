import numpy as np
import os
import utils
import pdb

debug = 0

#-------------------------------------------------------------------------------
## Input Parsing

trainObj = None
initObj = None
resFile = None
num_epochs = 0
learning_rate = 0

while( initObj is None ):
    initFile = raw_input("Enter the name of the initialization file: ")
    initObj = utils.parseFile(initFile, "weights")
    if( initObj is None ):
        print "Invalid weights file!\n"

while( trainObj is None ):
    trainFile = raw_input("Enter the name of the training data file: ")
    trainObj = utils.parseFile(trainFile, "data")
    if( trainObj is None ):
        print "Invalid data file!\n"

while( resFile is None ):
    resFileInput = raw_input("Enter the name of the trained weights output file: ")
#    if( os.access(resFileInput, os.W_OK) ):
    resFile = resFileInput 
#    else:
#        print "Invalid output file path!"

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


#------------------------------------------------------------------------------
## Initializing variables

in_layer = np.zeros( initObj[0][0]+1 )
hidden_layer = np.zeros( initObj[0][1]+1 )
out_layer = np.zeros( initObj[0][2] )

out_error = np.zeros( len(out_layer) )
hidden_error = np.zeros( len(hidden_layer)+1)

in2h_weights = initObj[1]
h2o_weights = initObj[2]

num_train = trainObj[0][0]
features_train = trainObj[1]
targets_train = trainObj[2]

#-----------------------------------------------------------------------------
## Back Propagation Learning

for epoch in range(0, num_epochs):
    for data in range(0, num_train):
        # Forward Propagation of training data, in -> out layer
        in_layer[1:] = features_train[data]
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
        if debug:
            print "Before Back prop"
            pdb.set_trace()
        
        
        # Back Propogation of errors, out -> in layer
        for out_node in range(0, len(out_layer)):
            out_error[out_node] = utils.sigmoidPrime( out_accumulator[out_node] )*(targets_train[data][out_node] - out_layer[out_node])
        if debug:
            print "Out_error calculated"
            pdb.set_trace()
        
        for hidden_node in range(0, len(hidden_layer)):
            hidden_error_accumulator = 0
            for out_node in range(0, len(out_layer)):
                hidden_error_accumulator += h2o_weights[out_node][hidden_node]*out_error[out_node]
            hidden_error[hidden_node] = utils.sigmoidPrime( hidden_accumulator[hidden_node] )*hidden_error_accumulator
        
        if debug:
            print "Before weight update"
            pdb.set_trace()
        
        # Update every weight using errors
        for hidden_node in range(1, len(hidden_layer)):
            for in_node in range(0, len(in_layer)):
                in2h_weights[hidden_node-1][in_node] = in2h_weights[hidden_node-1][in_node] + learning_rate*in_layer[in_node]*hidden_error[hidden_node]
        
        for out_node in range(0, len(out_layer)):
            for hidden_node in range(0, len(hidden_layer)):
                h2o_weights[out_node][hidden_node] = h2o_weights[out_node][hidden_node] + learning_rate*hidden_layer[hidden_node]*out_error[out_node]
        if debug:
            print "After weight update"
            pdb.set_trace()

#----------------------------------------------------------------------------
## Write the trained weights output file

with open(resFile, 'w') as file:
    file.write(' '.join('%s' % ii for ii in initObj[0]))
    for hidden_node in range(0, initObj[0][1]):
        file.write('\n' + ' '.join('%.3f' % ii for ii in in2h_weights[hidden_node]))
    for out_node in range(0, initObj[0][2]):
        file.write('\n' + ' '.join('%.3f' % ii for ii in h2o_weights[out_node]))


