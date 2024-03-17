import os
import json
import numpy as np

# define a function to generate the ML subsets
def generate_splits(classes, directory, split_file="../data/processed/split_definition.json"):
    # initialize empty lists to store train, validation, and test filenames
    train_files, val_files, test_files = [], [], []
    
    # check if split definition file exists
    if os.path.exists(split_file):
        print("train/validation/test subsets were loaded from a pre-generated file")
        # load split definition file
        with open(split_file, 'r') as file:
            split_data = json.load(file)
            train_files = split_data['Train']
            val_files = split_data['Validation']
            test_files = split_data['Test']
            
    else:
        print("train/validation/test subsets were generated and saved to a file")
        # iterate over each class
        for class_name in classes:
            # get the directory path for the current class
            class_dir = os.path.join(directory, class_name)
            # list all files in the directory
            files = os.listdir(class_dir)
            # shuffle the list of files
            np.random.shuffle(files)
            # calculate split points
            total_files = len(files)
            train_split = int(total_files * 0.6)
            val_split = int(total_files * 0.2)
            # assign files to train, validation, and test sets
            train_files.extend([(class_name, file) for file in files[:train_split]])
            val_files.extend([(class_name, file) for file in files[train_split:train_split+val_split]])
            test_files.extend([(class_name, file) for file in files[train_split+val_split:]])
            
        # shuffle the train, validation, and test sets
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)
        np.random.shuffle(test_files)
        
        # save split definition to a json file
        with open(split_file, 'w') as file:
            json.dump({'Train': train_files, 'Validation': val_files, 'Test': test_files}, file)
            
    # display the number of files in each set
    print("\tNumber of train files:", len(train_files))
    print("\tNumber of val files:", len(val_files))
    print("\tNumber of test files:", len(test_files))
    
    return train_files, val_files, test_files
