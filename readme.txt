
Terrain Classification Using Satellite Imagery

Apr 2024

Designed and implemented a computer vision system for automatic terrain classification using high-resolution multispectral imagery; leveraged variety of image analysis and feature extraction approaches to improve model performance.

Virtual environment details in ./requirements.txt.

Data Acquisition done using direct download.  Details in ./src/data/.  

Datasets (original) stored locally in ./data/raw.

Key references stored in ./references/literature/.

Data preprocessing and modeling done using Jupyter Notebooks.  Details in ./notebooks/. The notebooks included are:
    Primary Notebooks:
        *   jsm_feature_selection.ipynb: Choose which features need to be extracted, execution controlled by a set of flags.
        *   jsm_pca.ipynb: Read trained vector data from above stage and perform PCA. Visualize using tSNE. Pickle principal components
        *   jsm_classifiers.ipynb: Take principal components and run gridsearch. Pick optimal hyperparameters and train classifier. Evaluate using training and validation datasets.
        *   utils.py: Utility functions
    EDA/Supplementary Notebooks:
        -   jsm_eda_0.ipynb
        -   jsm_eda_1.ipynb
        -   mc_texture_features.ipynb
        -   mc_hsv_features.ipynb
        -   mm_freq_features.ipynb
        -   mm_sift_features.ipynb

Finalized tuned Models were pickled and are available in ./data/processed/

Final and interim reports in ./reports/.

Steps to run these notebooks and train the classifiers:

1) Download the PatternNet images from https://sites.google.com/view/zhouwx/dataset
2) Move the downloaded file (PatterNet.zip) to ./data/raw/PatternNet and untar in this directory
3) Open the jsm_feature_selection notebook. Select the feature flags that are of interest and set them to True: 
	INCLUDE_RGB_FEATURES = False
	INCLUDE_HSV_FEATURES = False
	INCLUDE_HOG_FEATURES = False
	INCLUDE_GLCMS_FEATURES = False
	INCLUDE_FFT_FEATURES = False
	INCLUDE_SIFT_FEATURES = False

4) Select a small subset of image on which features are extracted. Set this LIMIT_NUM_IMAGES to a non-zero value (100 preferably).

5) Run the jsm_feature_selection notebook. At the end of the run, all features are extracted and the relevant output files are archived in ./data/processed.

6) If you intend to run on the entire dataset, then set LIMIT_NUM_IMAGES = 0 and re-run

7) To perform PCA and visualize tSNE, run the jsm_pca.ipynb notebook. This will also pickle the principal components across all featuresets.

8) To train the classifiers and evaluate the trained models with validation/test datasets, run the jsm_classifier notebook. It will generate summary results and confusion matrix for the trainining, validation and test datasets





