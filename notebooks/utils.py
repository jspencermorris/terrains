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



def create_gaussian_filter(rows, cols, sigma):
    # Create Gaussian Filter: Low Pass Filter
    M,N = rows, cols
    H = np.zeros((M,N), dtype=np.float32)
    D0 = sigma
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))

    return H


def fft_image(image):

    # Read the image
    # image = plt.imread(image)

    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    ydim, xdim = image_gray.shape
    win = np.outer(np.hanning(ydim),np.hanning(xdim))
    win = win/np.mean(win)
    
    # Compute the 2D FFT of the grayscale image
    fft_image = np.fft.fft2(image_gray*win)

    # Shift the zero frequency component to the center
    fft_image_shifted = np.fft.fftshift(fft_image)

    # Compute the magnitude spectrum (absolute value) of the shifted FFT
    magnitude_spectrum = np.abs(fft_image_shifted)
    
    return magnitude_spectrum

def apply_gaussian_filter(image, filt):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image_gray = np.mean(image, axis=2)
    else:
        image_gray = image

    # Compute the 2D FFT of the image
    fft_image = np.fft.fft2(image_gray)
    fft_shifted = np.fft.fftshift(fft_image)

    # Apply the mask to the frequency domain representation
    fft_shifted_filtered = fft_shifted * filt
    
    # Compute the inverse 2D FFT
    ifft_shifted_filtered = np.fft.ifftshift(fft_shifted_filtered)
    filtered_image = np.fft.ifft2(ifft_shifted_filtered)
    filtered_image = np.abs(filtered_image)

    return filtered_image.astype(np.uint8)

def display_counts_per_class(filename, class_dict):
    sum = 0  
    for categ in class_dict:
        print(f'{categ}: {len(class_dict[categ])}')
        sum += len(class_dict[categ])
    print(f'Filename: {filename}  Total Count: {sum}')
    print()