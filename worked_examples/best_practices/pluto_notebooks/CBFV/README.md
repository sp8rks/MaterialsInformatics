# CBFV
Tool to quickly create a composition-based feature vector

## Usage
The file `example_code.py` can be used to quickly make predictions for any given property. Simply input the location of your desired train and test data. 

## Making the composition-based feature vector
The folder cbfv has the script `composition.py` and the folder "element_properties"
This script uses some chemical parsing tools from matminer and then does numpy operations to vectorize composition at a rate of ~10,000 formulae per second. 

## Getting a full model for train and test.
See `example_code.py` to featurize data and train models with the provided data

## Getting features for a given csv
`featurize_file.py` takes in the path of a csv files you want featurized and saves the file "X.csv" into the "featurized_data" folder
