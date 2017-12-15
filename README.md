Single Hidden Layer, Feed-Forward Neural Network
===============================================

A detailed description of the project goals can be found in `project_description.docx`. All commands are relative to the project root.

How to run, using Python2.7:
```
python train.py
python test.py
```

Structure
---------

The neural network is coded in Python. There are two main scripts, train.py and test.py. Utility functions are defined in utils.py. All data is organized in the datasets directory by name, with subdirectories for data, weights, and results.

For quick testing with varying input parameters (learning rate and epoch), change the values in `default_train.txt` then use the following commands:
```
cat default_train.txt | python train.py
cat default_test.txt | python test.py
cat datasets/drugs/result/results.txt
```

Special Dataset - Predicting drug use
-------------------------------------

I found my special dataset online. It is saved in this project in `datasets/drugs/data/drug_consumption.data`. A description of the dataset from the creators can be found in `datasets/drugs/data/data_description.txt`. The dataset has features based on an individual's personal characteristics like age, gender, and education level, as well personality scores based on NEO-FFI-R testing. The features were categorical, but were translated to quantative features by the creators. These features are used to predict an individual's usage rate of multiple drugs into 7 classes ranging from never used to used in the last day.

My function to create training and testing data as well as an initialization file from this dataset can be found in utils.py, named `generate_drug_data()`. The easiest way to call this function is:
```
python
import utils
utils.generate_drug_data()
```
One interesting thing about this dataset is that it attempts to find a mapping from the input features to the output usage rate for multiple drugs. My data generation function therefore takes as input the desired drug to base the neural network on. For my parameter testing, I used cannabis as my target drug since it had the most balanced distribution over the output classifier tags. Equal balance of training data classes is important to make sure that the neural network doesn't skew towards particular classes.

This function changes the input data by linearly scaling each feature into the range 0-1 by min and max values. It also changes the output CL# classifier tags into a 6-bit thermometer encoded binary output, similar to the grades dataset provided. This is somewhat reasonable, since it essentially encodes the output as a magnitude of drug use. If the data is simply left as a 1-hot encoding (similar to their classification scheme), the testing results were very poor.

The default parameters generate a dataset based on the drug cannabis with 9 hidden nodes in the initialization file. The initialization file is simply a valid network for the defined number of nodes in each layer, with a random number from the Uniform(0,1) distribution as the weight. I tested my neural network with many combinations of hidden nodes, learning rate, and number of epochs. After testing, I found that a decent parameter set was 9 hidden nodes with 100 epochs of training and 0.1 learning rate. I found that variation of these parameters within one order of magnitude caused little variation in the output micro and macro averaged results. Outside of this one order of magnitude range, the results seemed to suffer. However it seems like the small set of values in each of the categorical features may be limiting the effectiveness of varying the neural network parameters.


