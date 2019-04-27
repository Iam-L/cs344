"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO Topic Classification

Notes:

Proceeding with provided labeled SLO TBL dataset.  Will attempt to preprocess and train this.

Using the "NLTK" Natural Language Toolkit as replacement for CMU Tweet Tagger preprocessor.

Resources Used:

https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan
(drop rows with NaN values in columns)

https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
(pandas row iteration methods)

https://stackoverflow.com/questions/40408471/select-data-when-specific-columns-have-null-value-in-pandas
(create boolean indexing mask)

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
(drop pandas columns)

https://stackoverflow.com/questions/12850345/how-to-combine-two-data-frames-in-python-pandas
(combine dataframes)

https://stackoverflow.com/questions/23667369/drop-all-duplicate-rows-in-python-pandas
(drop duplicate examples)

https://www.nltk.org/
(text pre-processing)

https://stackoverflow.com/questions/34784004/python-text-processing-nltk-and-pandas
(tokenize tweets using pands and nltk)

"""

################################################################################################################

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk as nltk

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
print("The shape of our SLO dataframe:")
print(slo_dataframe.shape)
print()
print("The columns of our SLO dataframe:")
print(slo_dataframe.head)
print()

# Assign column names.
slo_dataframe.columns = column_names

################################################################################################################

# Data pre-processing
# TODO - use pre-processing methods indicated in SLO article
# TODO - https://github.com/Calvin-CS/slo-classifiers/tree/feature/keras-nn/stance/data
# TODO - https://github.com/Calvin-CS/slo-classifiers/blob/feature/keras-nn/stance/data/tweet_preprocessor.py
# FIXME - preprocessor will only work on Linux/Mac

# Drop all rows with only NaN in all columns.
slo_dataframe = slo_dataframe.dropna(how='all')
# Drop all rows without at least 2 non NaN values - indicating no SLO TBL classification labels.
slo_dataframe = slo_dataframe.dropna(thresh=2)

print(slo_dataframe.shape)
print()

if debug:
    # Iterate through each row and check we dropped properly.
    print()
    print("Dataframe only with examples that have SLO TBL classification labels:")
    for index in slo_dataframe.index:
        print(slo_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo_dataframe['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe['SLO2'][index]) + '\tSLO3: ' + str(slo_dataframe['SLO3'][index]))
    print("Shape of dataframe with SLO TBL classifications: " + str(slo_dataframe.shape))

#######################################################

# Boolean indexing to select examples with only a single SLO TBL classification.
mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].isna() & slo_dataframe['SLO3'].isna())

# Check that boolean indexing is working.
print()
print("Check that our boolean indexing mask gives only examples with a single SLO TBL classifications:")
print(mask.tail)
print("The shape of our boolean indexing mask:")
print(mask.shape)

# Create new dataframe with only those examples with a single SLO TBL classification.
slo_dataframe_single_classification = slo_dataframe[mask]

# Check that we have created the new dataframe properly.
if debug:
    # Iterate through each row and check that only examples with multiple SLO TBL classifications are left.
    print("Dataframe only with examples that have a single SLO TBL classification label:")
    for index in slo_dataframe_single_classification.index:
        print(slo_dataframe_single_classification['Tweet'][index]
              + '\tSLO1: ' + str(slo_dataframe_single_classification['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe_single_classification['SLO2'][index])
              + '\tSLO3: ' + str(slo_dataframe_single_classification['SLO3'][index]))
    print("Shape of dataframe with a single SLO TBL classification: "
          + str(slo_dataframe_single_classification.shape))

#######################################################

# Drop SLO2 and SLO3 columns as they are just NaN values.
slo_dataframe_single_classification = slo_dataframe_single_classification.drop(columns=['SLO2', 'SLO3'])

if debug:
    print('\n')
    print("Dataframe with SLOW2 and SLO3 columns dropped as they are just NaN values:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo_dataframe_single_classification.index:
        print(slo_dataframe_single_classification['Tweet'][index] + '\tSLO1: '
              + str(slo_dataframe_single_classification['SLO1'][index]))
    print("Shape of slo_dataframe_single_classification: " + str(slo_dataframe_single_classification.shape))

# Re-name columns.
column_names_single = ['Tweet', 'SLO']

slo_dataframe_single_classification.columns = column_names_single

#######################################################

# Boolean indexing to select examples with multiple SLO TBL classifications.
mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].notna() | slo_dataframe['SLO3'].notna())

# Check that boolean indexing is working.
print()
print("Check that our boolean indexing mask gives only examples with multiple SLO TBL classifications:")
print(mask.tail)
print("The shape of our boolean indexing mask:")
print(mask.shape)

# Create new dataframe with only those examples with multiple SLO TBL classifications.
slo_dataframe_multiple_classifications = slo_dataframe[mask]

# Check that we have created the new dataframe properly.
if debug:
    # Iterate through each row and check that only examples with multiple SLO TBL classifications are left.
    print("Dataframe only with examples that have multiple SLO TBL classification labels:")
    for index in slo_dataframe_multiple_classifications.index:
        print(slo_dataframe_multiple_classifications['Tweet'][index]
              + '\tSLO1: ' + str(slo_dataframe_multiple_classifications['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe_multiple_classifications['SLO2'][index])
              + '\tSLO3: ' + str(slo_dataframe_multiple_classifications['SLO3'][index]))
    print("Shape of dataframe with multiple SLO TBL classifications: "
          + str(slo_dataframe_multiple_classifications.shape))

#######################################################

# Duplicate examples with multiple SLO TBL classifications into examples with only 1 SLO TBL classification each.
slo1_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO2', 'SLO3'])
slo2_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO3'])
slo3_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO2'])

if debug:
    print('\n')
    print("Separated dataframes single label for examples with multiple SLO TBL classification labels:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo1_dataframe.index:
        print(slo1_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo1_dataframe['SLO1'][index]))
    for index in slo2_dataframe.index:
        print(slo2_dataframe['Tweet'][index] + '\tSLO2: ' + str(slo2_dataframe['SLO2'][index]))
    for index in slo3_dataframe.index:
        print(slo3_dataframe['Tweet'][index] + '\tSLO3: ' + str(slo3_dataframe['SLO3'][index]))
    print("Shape of slo1_dataframe: " + str(slo1_dataframe.shape))
    print("Shape of slo2_dataframe: " + str(slo2_dataframe.shape))
    print("Shape of slo3_dataframe: " + str(slo3_dataframe.shape))

# Re-name columns.
column_names_single = ['Tweet', 'SLO']

slo1_dataframe.columns = column_names_single
slo2_dataframe.columns = column_names_single
slo3_dataframe.columns = column_names_single

#######################################################

# Concatenate the individual dataframes back together.
frames = [slo1_dataframe, slo2_dataframe, slo3_dataframe, slo_dataframe_single_classification]
slo_dataframe_combined = pd.concat(frames, ignore_index=True)

# Note: Doing this as context-sensitive menu stopped displaying all useable function calls after concat.
slo_dataframe_combined = pd.DataFrame(slo_dataframe_combined)

if debug:
    print('\n')
    print("Recombined individual dataframes for the dataframe representing Tweets with only a single SLO TBL "
          "classification example\n and for the dataframes representing Tweets with multiple SLO TBL classification "
          "labels:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo_dataframe_combined.index:
        print(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
    print('Shape of recombined dataframes: ' + str(slo_dataframe_combined.shape))

#######################################################

# Drop all rows with only NaN in all columns.
slo_dataframe_combined = slo_dataframe_combined.dropna()

if debug:
    print('\n')
    print("Recombined dataframes - NaN examples removed:")
    # Iterate through each row and check that we no longer have examples with NaN values.
    for index in slo_dataframe_combined.index:
        print(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
    print('Shape of recombined dataframes without NaN examples: ' + str(slo_dataframe_combined.shape))

#######################################################

# Drop duplicate examples with the same SLO TBL classification values.
slo_dataframe_TBL_duplicates_dropped = slo_dataframe_combined.drop_duplicates(subset=['Tweet', 'SLO'], keep=False)

if debug:
    print('\n')
    print("Same examples with duplicate SLO TBL classifications removed:")
    # Iterate through each row and check that we no longer have examples with NaN values.
    for index in slo_dataframe_TBL_duplicates_dropped.index:
        print(slo_dataframe_TBL_duplicates_dropped['Tweet'][index] + '\tSLO: '
              + str(slo_dataframe_TBL_duplicates_dropped['SLO'][index]))
    print('Shape of dataframes without duplicate TBL values: ' + str(slo_dataframe_TBL_duplicates_dropped.shape))

################################################################################################################

# Create input features.
selected_features = slo_dataframe_TBL_duplicates_dropped[column_names_single]
processed_features = selected_features.copy()

# Check what we are using for input features.
if debug:
    print()
    print("The tweets as a string:")
    print(processed_features['Tweet'])
    print()
    print("SLO TBL classification:")
    print(processed_features['SLO'])

############################################################################################

"""
Main function.  Execute the program.
"""
# Debug variable.
debug_main = 0

if __name__ == '__main__':
    print()

############################################################################################
