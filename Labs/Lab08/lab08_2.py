"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 08 - Feature Engineering

Notes:

Exercise 8.2 - Feature Crosses

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
from tensorflow.python.feature_column.feature_column_v2 import crossed_column

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
        feature_columns,
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
      feature_columns: A `set` specifying the input feature columns to use.
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
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

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
            steps=steps_per_period
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

def get_quantile_based_boundaries(feature_values, num_buckets):
    """
    Functions computes boundaries based on quantiles, so each bucket contains equal # of elements.

    :param feature_values: self-explanatory
    :param num_buckets: # of buckets to create.
    :return:  bucketized values.
    """
    boundaries = np.arange(1.0, num_buckets) / num_buckets
    quantiles = feature_values.quantile(boundaries)
    return [quantiles[q] for q in quantiles.keys()]


###########################################################################################

def construct_feature_columns_bucketized():
    """
    Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            training_examples["households"], 7))

    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    #
    # YOUR CODE HERE: bucketize the following columns, following the example above:
    #

    # Divide latitude into 10 buckets.
    bucketized_latitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    # Divide housing_median_age into 10 buckets.
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    # Divide median_income into 10 buckets.
    bucketized_median_income = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    # Divide rooms_per_person into 10 buckets.
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    feature_columns = {bucketized_longitude, bucketized_latitude, bucketized_housing_median_age, bucketized_households,
                       bucketized_median_income, bucketized_rooms_per_person}

    return feature_columns

###########################################################################################

def construct_feature_columns_crossed():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """
    households = tf.feature_column.numeric_column("households")
    longitude = tf.feature_column.numeric_column("longitude")
    latitude = tf.feature_column.numeric_column("latitude")
    housing_median_age = tf.feature_column.numeric_column("housing_median_age")
    median_income = tf.feature_column.numeric_column("median_income")
    rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")

    # Divide households into 7 buckets.
    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            training_examples["households"], 7))

    # Divide longitude into 10 buckets.
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            training_examples["longitude"], 10))

    # Divide latitude into 10 buckets.
    bucketized_latitude = tf.feature_column.bucketized_column(
        latitude, boundaries=get_quantile_based_boundaries(
            training_examples["latitude"], 10))

    # Divide housing_median_age into 7 buckets.
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        housing_median_age, boundaries=get_quantile_based_boundaries(
            training_examples["housing_median_age"], 7))

    # Divide median_income into 7 buckets.
    bucketized_median_income = tf.feature_column.bucketized_column(
        median_income, boundaries=get_quantile_based_boundaries(
            training_examples["median_income"], 7))

    # Divide rooms_per_person into 7 buckets.
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        rooms_per_person, boundaries=get_quantile_based_boundaries(
            training_examples["rooms_per_person"], 7))

    # YOUR CODE HERE: Make a feature column for the long_x_lat feature cross
    # long_x_lat = crossed_column(['longitude', 'latitude'], 1000)

    long_x_lat = tf.feature_column.crossed_column(
        {bucketized_longitude, bucketized_latitude}, hash_bucket_size=1000)

    feature_columns = {bucketized_longitude, bucketized_latitude, bucketized_housing_median_age, bucketized_households,
                       bucketized_median_income, bucketized_rooms_per_person, long_x_lat}

    return feature_columns

###########################################################################################

"""
Main function. Executes the program.
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

    # Train the model.
    # _ = train_model(
    #     learning_rate=1.0,
    #     steps=500,
    #     batch_size=100,
    #     feature_columns=construct_feature_columns(training_examples),
    #     training_examples=training_examples,
    #     training_targets=training_targets,
    #     validation_examples=validation_examples,
    #     validation_targets=validation_targets)

    # Divide households into 7 buckets.
    households = tf.feature_column.numeric_column("households")
    bucketized_households = tf.feature_column.bucketized_column(
        households, boundaries=get_quantile_based_boundaries(
            california_housing_dataframe["households"], 7))

    # Divide longitude into 10 buckets.
    longitude = tf.feature_column.numeric_column("longitude")
    bucketized_longitude = tf.feature_column.bucketized_column(
        longitude, boundaries=get_quantile_based_boundaries(
            california_housing_dataframe["longitude"], 10))

    """
    Task 1: Train the Model on Bucketized Feature Columns
    
    Refer to function "def construct_feature_columns_bucketized():" above.
    
    Note: Stuck with 10 buckets for each bucketized feature columns.
    """
    # Train the model.
    # _ = train_model(
    #     learning_rate=1.0,
    #     steps=500,
    #     batch_size=100,
    #     feature_columns=construct_feature_columns_bucketized(),
    #     training_examples=training_examples,
    #     training_targets=training_targets,
    #     validation_examples=validation_examples,
    #     validation_targets=validation_targets)

    """
    Task 2: Train the Model Using Feature Crosses
    
    Refer to function "def construct_feature_columns_crossed():" above.
    
    Note: Did it wrong the first time.  Commented out the incorrect method.  Was wondering why it was taking forever
    it train the data.  Looked at the solution for the correct way.
    
    URL: https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column
    """
    # Train the model.
    _ = train_model(
        learning_rate=1.0,
        steps=500,
        batch_size=100,
        feature_columns=construct_feature_columns_crossed(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

###########################################################################################

"""
Exercise 8.2 Questions:
###########################################################################################

They recommend FTRL for L1 optimization, but the code specifies the same rate (learning_rate) for all runs. 
How is FTRL managing the learning rate?

__init__(
    learning_rate,
    learning_rate_power=-0.5,
    initial_accumulator_value=0.1,
    l1_regularization_strength=0.0,
    l2_regularization_strength=0.0,
    use_locking=False,
    name='Ftrl',
    accum_name=None,
    linear_name=None,
    l2_shrinkage_regularization_strength=0.0
)

learning_rate: A float value or a constant float Tensor.

learning_rate_power: A float value, must be less or equal to zero. 
Controls how the learning rate decreases during training. 
Use zero for a fixed learning rate. See section 3.1 in the paper.

So apparently, tf.train.FtrlOptimizer dynamically decreases the learning rate during training for you as long as you
specify a non-zero value for learning_rate_power.

########################################################

What good does the bucketing/binning do?

Referring below, it's definitely lowering RMSE drastically versus using standard features or other synthetic features.

Binning effectively deals with the clustering of Los Angeles and San Francisco at the specific latitude/longitudes 
those cities are located at where housing is concentrated.

########################################################

Submit your solutions to tasks 1–2. Did you find their task 1 bucketing to make sense? 
Identify one unique feature cross for task 2 and explain how it could be useful.

The task 1 bucketing made sense though I'm not sure how exactly you determine the # of buckets to separate each
feature column into when making the bucketized feature columns.

I would cross median_income and total_rooms to see if there is a relationship between the # of rooms in the house
based on income.  Maybe higher income = more rooms and vice versa.  Maybe there is a more complex relationship.

########################################################
Task 1 Training Output:

Training model...
RMSE (on training data):
  period 00 : 206.20
  period 01 : 192.41
  period 02 : 182.18
  period 03 : 173.86
  period 04 : 166.77
  period 05 : 160.61
  period 06 : 155.13
  period 07 : 150.25
  period 08 : 145.86
  period 09 : 141.90
Model training finished.

Process finished with exit code 0

########################################################
Task 2 Training Output:

Training model...
RMSE (on training data):
  period 00 : 162.21
  period 01 : 134.15
  period 02 : 117.29
  period 03 : 106.21
  period 04 : 98.36
  period 05 : 92.68
  period 06 : 88.35
  period 07 : 84.99
  period 08 : 82.24
  period 09 : 79.89
Model training finished.

Process finished with exit code 0

########################################################
Other console  output:

D:\Dropbox\cs344-ai\venv3.6-64bit\Scripts\python.exe D:/Dropbox/cs344-ai/cs344/Labs/Lab08/lab08_2.py
Training examples summary:
       latitude  longitude  ...  median_income  rooms_per_person
count   12000.0    12000.0  ...        12000.0           12000.0
mean       35.6     -119.6  ...            3.9               2.0
std         2.2        2.0  ...            1.9               1.2
min        32.5     -124.3  ...            0.5               0.0
25%        33.9     -121.8  ...            2.6               1.5
50%        34.2     -118.5  ...            3.5               1.9
75%        37.7     -118.0  ...            4.8               2.3
max        42.0     -114.3  ...           15.0              55.2

[8 rows x 9 columns]
Validation examples summary:
       latitude  longitude  ...  median_income  rooms_per_person
count    5000.0     5000.0  ...         5000.0            5000.0
mean       35.6     -119.5  ...            3.9               2.0
std         2.1        2.0  ...            1.9               1.0
min        32.6     -124.2  ...            0.5               0.1
25%        33.9     -121.8  ...            2.5               1.5
50%        34.2     -118.5  ...            3.5               1.9
75%        37.7     -118.0  ...            4.8               2.3
max        41.9     -114.5  ...           15.0              34.2

[8 rows x 9 columns]
Training targets summary:
       median_house_value
count             12000.0
mean                207.1
std                 116.0
min                  15.0
25%                 118.9
50%                 179.9
75%                 265.0
max                 500.0
Validation targets summary:
       median_house_value
count              5000.0
mean                207.9
std                 115.9
min                  17.5
25%                 120.9
50%                 181.2
75%                 265.2
max                 500.0

###########################################################################################

Submit your solutions to exercises 1–2.
"""
