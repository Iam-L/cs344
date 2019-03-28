"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 08 - Feature Engineering

Notes:

Exercise 8.1 - Feature Sets

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.
"""

###########################################################################################

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))


###########################################################################################

def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
    selected_features = california_housing_dataframe[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
            california_housing_dataframe["total_rooms"] /
            california_housing_dataframe["population"])
    return processed_features


###########################################################################################

def preprocess_targets(california_housing_dataframe):
    """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
            california_housing_dataframe["median_house_value"] / 1000.0)
    return output_targets


###########################################################################################

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


###########################################################################################

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


###########################################################################################

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    return linear_regressor


###########################################################################################

def select_and_transform_features(source_df):
    """
    Task 2:

    Function performs binning, "bucketing" of latitude versus median_income.

    :param source_df: the source dataframe to bin.
    :return:  the binned dataframe.
    """
    LATITUDE_RANGES = zip(range(32, 44), range(33, 45))

    selected_examples = pd.DataFrame()
    selected_examples["median_income"] = source_df["median_income"]

    for r in LATITUDE_RANGES:
        selected_examples["latitude_%d_to_%d" % r] = source_df["latitude"].apply(
            lambda l: 1.0 if l >= r[0] and l < r[1] else 0.0)
    return selected_examples


###########################################################################################

def select_and_transform_features2(source_df):
    """
    Function creates a synthetic feature.

    :param source_df: the source dataframe.
    :return:  the synthetic feature.
    """
    processed_features = source_df.copy()
    # Create a synthetic feature.
    processed_features["median_income_per_rooms_per_person"] = (
            source_df["median_income"] /
            source_df["rooms_per_person"])
    return processed_features


###########################################################################################

"""
Main function. Executes program.
"""
if __name__ == '__main__':
    # Choose the first 12000 (out of 17000) examples for training.
    training_examples = preprocess_features(california_housing_dataframe.head(12000))
    training_targets = preprocess_targets(california_housing_dataframe.head(12000))

    # Choose the last 5000 (out of 17000) examples for validation.
    validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

    # Double-check that we've done the right thing.
    print("Training examples summary:")
    display.display(training_examples.describe())
    print("Validation examples summary:")
    display.display(validation_examples.describe())

    print("Training targets summary:")
    display.display(training_targets.describe())
    print("Validation targets summary:")
    display.display(validation_targets.describe())

    """
    Task 1: Develop a Good Feature Set
    
    A correlation matrix shows pairwise correlations, both for each feature compared to the target and 
    for each feature compared to other features.

    Here, correlation is defined as the Pearson correlation coefficient. You don't have to understand 
    the mathematical details for this exercise.

    Correlation values have the following meanings:
    
        -1.0: perfect negative correlation
        0.0: no correlation
        1.0: perfect positive correlation
    """
    # Construct correlation dataframe.
    correlation_dataframe = training_examples.copy()
    correlation_dataframe["target"] = training_targets["median_house_value"]

    print(correlation_dataframe.corr())

    #
    # Your code here: add your features of choice as a list of quoted strings.
    #
    minimal_features = ["median_income", "rooms_per_person"]

    assert minimal_features, "You must select at least one feature!"

    minimal_training_examples = training_examples[minimal_features]
    minimal_validation_examples = validation_examples[minimal_features]

    #
    # Don't forget to adjust these parameters.
    #
    # train_model(
    #     learning_rate=0.0001,
    #     steps=250,
    #     batch_size=10,
    #     training_examples=minimal_training_examples,
    #     training_targets=training_targets,
    #     validation_examples=minimal_validation_examples,
    #     validation_targets=validation_targets)

    # Scatter-plot graph.
    plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
    plt.show()

    """
    Task 2: Make Better Use of Latitude
    
    Note: I did look at the solution.  I had no idea how to implement binning.
    
    Refer to function "def select_and_transform_features(source_df):" above.
    """
    #
    # YOUR CODE HERE: Train on a new data set that includes synthetic features based on latitude.
    #
    selected_training_examples = select_and_transform_features(training_examples)
    selected_validation_examples = select_and_transform_features(validation_examples)

    #
    # Don't forget to adjust these parameters.
    #
    train_model(
        learning_rate=0.001,
        steps=1000,
        batch_size=100,
        training_examples=selected_training_examples,
        training_targets=training_targets,
        validation_examples=selected_validation_examples,
        validation_targets=validation_targets)

###########################################################################################

"""
Exercise 8.1 Questions:
###########################################################################################

What does the Pearson correlation coefficient measure? 

A measure of the linear correlation between two variables X and Y.

########################################################

Identify one example value from the correlation table you compute and explain why it makes sense.

"median_income" has a high correlation with the target "median_housing_value" because the more somebody makes,
the more likely they can afford to buy a more expensive house.

########################################################
Output from chosen minimal_features for Task 1:

Training model...
RMSE (on training data):
  period 00 : 236.22
  period 01 : 236.17
  period 02 : 236.11
  period 03 : 236.06
  period 04 : 236.00
  period 05 : 235.94
  period 06 : 235.89
  period 07 : 235.83
  period 08 : 235.77
  period 09 : 235.72
Model training finished.

Process finished with exit code 0

########################################################
Output from Task 2:

Training model...
RMSE (on training data):
  period 00 : 236.12
  period 01 : 234.04
  period 02 : 231.96
  period 03 : 229.88
  period 04 : 227.81
  period 05 : 225.74
  period 06 : 223.67
  period 07 : 221.61
  period 08 : 219.55
  period 09 : 217.50
Model training finished.

Process finished with exit code 0

########################################################

Submit your solutions to tasks 1–2. 

Include the features you selected for task 1 and the synthetic features you developed for task 2; 
include the final RMS errors but not the training output. Did you beat the Google-provided baselines?

No, I didn't beat the Google-provided baselines.  If I had another 50 periods of training, I might have beat it.

Again, it's time consuming to wait a minute for every training run.  So, I make a few attempts at most.

########################################################
Other console output:

D:\Dropbox\cs344-ai\venv3.6-64bit\Scripts\python.exe D:/Dropbox/cs344-ai/cs344/Labs/Lab08/lab08_1.py
Training examples summary:
       latitude  longitude  ...  median_income  rooms_per_person
count   12000.0    12000.0  ...        12000.0           12000.0
mean       35.6     -119.5  ...            3.9               2.0
std         2.1        2.0  ...            1.9               1.2
min        32.5     -124.3  ...            0.5               0.0
25%        33.9     -121.8  ...            2.6               1.5
50%        34.2     -118.5  ...            3.5               1.9
75%        37.7     -118.0  ...            4.8               2.3
max        42.0     -114.3  ...           15.0              55.2

[8 rows x 9 columns]
Validation examples summary:
       latitude  longitude  ...  median_income  rooms_per_person
count    5000.0     5000.0  ...         5000.0            5000.0
mean       35.7     -119.6  ...            3.8               2.0
std         2.1        2.0  ...            1.8               1.1
min        32.5     -124.3  ...            0.5               0.2
25%        33.9     -121.8  ...            2.6               1.5
50%        34.3     -118.6  ...            3.5               2.0
75%        37.7     -118.0  ...            4.8               2.3
max        41.9     -114.6  ...           15.0              41.3

[8 rows x 9 columns]
Training targets summary:
       median_house_value
count             12000.0
mean                207.8
std                 116.4
min                  15.0
25%                 120.2
50%                 181.1
75%                 265.0
max                 500.0
Validation targets summary:
       median_house_value
count              5000.0
mean                206.0
std                 115.0
min                  15.0
25%                 118.5
50%                 178.1
75%                 265.0
max                 500.0
                    latitude  longitude  ...  rooms_per_person  target
latitude                 1.0       -0.9  ...               0.1    -0.1
longitude               -0.9        1.0  ...              -0.1    -0.0
housing_median_age       0.0       -0.1  ...              -0.1     0.1
total_rooms             -0.0        0.1  ...               0.1     0.1
total_bedrooms          -0.1        0.1  ...               0.1     0.0
population              -0.1        0.1  ...              -0.1    -0.0
households              -0.1        0.1  ...              -0.0     0.1
median_income           -0.1       -0.0  ...               0.2     0.7
rooms_per_person         0.1       -0.1  ...               1.0     0.2
target                  -0.1       -0.0  ...               0.2     1.0

[10 rows x 10 columns]

###########################################################################################

Submit your solutions to exercises 1–2.
"""
