# Function to parse the input files
import numpy as np
import scipy.special
import csv

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
                features[lineNum] = thisLine[0:-fileParams[2]]
                targets[lineNum] = thisLine[-fileParams[2]:]
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

def accuracy( contingency ):
    return (contingency[0]+contingency[3])/sum(contingency)

def precision( contingency ):
    return contingency[0]/(contingency[0]+contingency[1])

def recall( contingency ):
    return contingency[0]/(contingency[0]+contingency[2])

def f1( contingency ):
    return 2*precision(contingency)*recall(contingency)/(precision(contingency)+recall(contingency))

def getTrainStats( contingency ):
    res = np.zeros(4)
    res[0] = accuracy(contingency)
    res[1] = precision(contingency)
    res[2] = recall(contingency)
    res[3] = f1(contingency)
    return res

def generate_drug_data( drug = "cannabis",
                        num_hidden_nodes = 9,
                        raw_data_file = "datasets/drugs/data/drug_consumption.data",
                        out_init = "datasets/drugs/weights/init.txt",
                        out_train_data = "datasets/drugs/data/train.txt",
                        out_test_data = "datasets/drugs/data/test.txt"
                        ):
    valid_drugs = ["alcohol", "amphetamine", "amyl_nitrite", "benzodiazepine", "caffeine", "cannabis", "chocalate", "cocaine", "crack", "ecstasy", "heroin", "ketamine", "legal_highs", "lsd", "meth", "mushrooms", "nicotine", "semeron", "volatile"]
    while not drug in valid_drugs:
        print "Error: invalid drug input. The valid drugs are: "
        print '\n'.join(valid_drugs)
        drug = raw_input("Enter a drug from the list above: ")
    
    labels = []
    data = []
    try:
        with open(raw_data_file, 'r') as file:
            reader = csv.reader(file, dialect='excel')
            for line in reader:
                data.append(list(map(float, line[1:13])))
                class_string = line[13+valid_drugs.index(drug)]
                labels.append(int(class_string[-1:]))
    except IOError:
        print "Error: invalid raw_data_file. Exiting the function."
    
    features = np.array(data)
    targets = np.array(labels)
    
    num_data = np.size(features,0)
    num_train = 1000
    target_binary = np.zeros( (num_data, 6) )

    for feature_num in range(0,12):
        min_value = np.min(features[:,feature_num])
        for data in range(0,num_data):
            features[data, feature_num] -= min_value
        max_value = np.max(features[:,feature_num])
        for data in range(0,num_data):
            features[data,feature_num] /= max_value
    
    for data in range(0, num_data):
        for bin_num in range(0, 6):
            if( targets[data] > bin_num ):
                target_binary[data, bin_num] = 1
    
    ## Write the train output file
    with open(out_train_data, 'w') as file:
        file.write(str(num_train)+" 12 6\n")
        for data in range(0, num_train):
            features_string = ' '.join('%.3f' % ii for ii in features[data])
            target_string = ' '.join('%d' % ii for ii in target_binary[data])
            this_line = features_string + ' ' + target_string + '\n'
            file.write(this_line)
    
    # Write the test output file
    with open(out_test_data, 'w') as file:
        file.write(str(num_data-num_train)+" 12 6\n")
        for data in range(num_train, num_data):
            features_string = ' '.join('%.3f' % ii for ii in features[data])
            target_string = ' '.join('%d' % ii for ii in target_binary[data])
            this_line = features_string + ' ' + target_string + '\n'
            file.write(this_line)
    
    # Write the init output file
    with open(out_init, 'w') as file:
        file.write("12 "+str(num_hidden_nodes)+" 6\n")
        for hidden_nodes in range(0,num_hidden_nodes):
            values = np.random.uniform(size=13)
            file.write(' '.join('%.3f' % ii for ii in values) + '\n')
        for out_nodes in range(0,6):
            values = np.random.uniform(size=num_hidden_nodes+1)
            file.write(' '.join('%.3f' % ii for ii in values) + '\n')
    
