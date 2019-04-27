"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO Topic Classification

Notes:

TODO - placeholder

Resources Used:

TODO - placeholder

"""

################################################################################################################

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

debug = True
################################################################################################################

# Import the dataset.
slo_dataset = \
    pd.read_csv("datasets/tbl_training_set.csv", sep=",")

# Shuffle the data randomly.
slo_dataset = slo_dataset.reindex(
    np.random.permutation(slo_dataset.index))

# Rename columns to something that makes sense.
column_names = ['Tweet', 'SLO1', 'SLO2', 'SLO3']

# Generate a Pandas dataframe.
slo_dataframe = pd.DataFrame(slo_dataset)

# Print shape and column names.
print("The shape of our SLO dataset:")
print(slo_dataframe.shape)
print()
print("The columns of our SLO dataset:")
print(slo_dataframe.head)
print()

# Assign column names.
slo_dataframe.columns = column_names

# Data pre-processing
# TODO - use pre-processing methods indicated in SLO article
# TODO - https://github.com/Calvin-CS/slo-classifiers/tree/feature/keras-nn/stance/data
# TODO - https://github.com/Calvin-CS/slo-classifiers/blob/feature/keras-nn/stance/data/tweet_preprocessor.py
# FIXME - preprocessor will only work on Linux/Mac

# Drop all rows with only NaN in all columns.
slo_dataframe = slo_dataframe.dropna(how='all')
# Drop all rows without at least 2 non NaN values - indicating no SLO classification labels.
slo_dataframe = slo_dataframe.dropna(thresh=2)
print(slo_dataframe.shape)
print()

if debug:
    # Iterate through each row.
    for index in slo_dataframe.index:
        print(slo_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo_dataframe['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe['SLO2'][index]) + '\tSLO3: ' + str(slo_dataframe['SLO3'][index]))

# Create input features.
selected_features = slo_dataframe[column_names]
processed_features = selected_features.copy()

print()
print("The tweets as a string:")
print(processed_features['Tweet'])
print()
print("SLO classification tag 1 (if any):")
print(processed_features['SLO1'])
print()
print("SLO classification tag 2 (if any):")
print(processed_features['SLO2'])
print()
print("SLO classification tag 3 (if any):")
print(processed_features['SLO3'])
print()

############################################################################################

"""
Main function.  Execute the program.
"""
# Debug variable.
debug = 0

if __name__ == '__main__':
    print()

############################################################################################
