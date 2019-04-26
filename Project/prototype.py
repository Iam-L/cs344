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

################################################################################################################

# Import the dataset.
slo_dataset = \
    pd.read_csv("datasets/tbl_training_set.csv", sep=",")

# Shuffle the data randomly.
slo_dataset = slo_dataset.reindex(
    np.random.permutation(slo_dataset.index))

# Rename columns to something that makes sense.
column_names = ['Tweet', 'SLO1', 'SLO2', 'SLO3']
slo_dataset.columns = column_names

# Print shape and column names.
print("The shape of our SLO dataset:")
print(slo_dataset.shape)
print()
print("The columns of our SLO dataset:")
print(slo_dataset.head)
print()

# Create input features.
selected_features = slo_dataset[column_names]
processed_features = selected_features.copy()

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

# Data pre-processing
# TODO - use pre-processing methods indicated in SLO article
# TODO - can I get the code that does the pre-processing and appropriate it for my own use here?

# TODO - https://github.com/Calvin-CS/slo-classifiers/tree/feature/keras-nn/stance/data
# TODO - https://github.com/Calvin-CS/slo-classifiers/blob/feature/keras-nn/stance/data/tweet_preprocessor.py

############################################################################################

"""
Main function.  Execute the program.
"""
# Debug variable.
debug = 0

if __name__ == '__main__':
    print()

############################################################################################

