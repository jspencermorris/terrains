# import libraries
import os
import zipfile

# PatternNet data sourced from: https://sites.google.com/view/zhouwx/dataset
# data was manually downloaded to ./data/raw/PatternNet.zip
zip_file_path = '../../data/raw/PatternNet.zip'

# create the directory if it doesn't already exist
os.makedirs('../../data/interim', exist_ok=True)

# extract the contents to the specified directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('../../data/interim')