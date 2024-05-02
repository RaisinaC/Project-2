Project-2 README

Aim

The aim of this project is to predict the maximum temperature of the air temperature at 2 meters above the ground (TREFMXAV_U) in Manchester from January 1, 2050, to December 31, 2080. The dataset consists of six files containing relevant weather data.

Files

data_loader.py: This script is responsible for loading and merging the six data files, filtering out data for Manchester, and performing necessary data preprocessing steps.
split_training_testing_by_time.py: This script separates the dataset into training and testing data based on time.
adding_seasonality.py: This script adds seasonality as one of the features to the dataset.
models.py: This script contains implementations of various machine learning models that will be used for prediction.
Main.ipynb: This Jupyter notebook serves as the main working document for the project. It contains the code for data loading, preprocessing, model training, evaluation, and prediction.

Usage

Run data_loader.py to load, merge, and preprocess the data.
Execute split_training_testing_by_time.py to split the dataset into training and testing sets.
Run adding_seasonality.py to add seasonality features to the dataset.
Use models.py to train and test different machine learning models for prediction.
Refer to Main.ipynb for a comprehensive overview and execution of the project workflow.

Dependencies:

Refer to the environment.yml file.
