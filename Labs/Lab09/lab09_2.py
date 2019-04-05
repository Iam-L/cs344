"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 09 - Classification

Notes:

Exercise 9.2 - Sparsity and L1 Regularization

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.
"""

###########################################################################################

from __future__ import print_function

import time

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
    # Create a boolean categorical feature representing whether the
    # median_house_value is above a set threshold.
    output_targets["median_house_value_is_high"] = (
            california_housing_dataframe["median_house_value"] > 265000).astype(float)
    return output_targets


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

def get_quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i + 1.) / (num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]


###########################################################################################

def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

    bucketized_households = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("households"),
        boundaries=get_quantile_based_buckets(training_examples["households"], 10))
    bucketized_longitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("longitude"),
        boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
    bucketized_latitude = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("latitude"),
        boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("housing_median_age"),
        boundaries=get_quantile_based_buckets(
            training_examples["housing_median_age"], 10))
    bucketized_total_rooms = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("total_rooms"),
        boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
    bucketized_total_bedrooms = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("total_bedrooms"),
        boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
    bucketized_population = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("population"),
        boundaries=get_quantile_based_buckets(training_examples["population"], 10))
    bucketized_median_income = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("median_income"),
        boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
    bucketized_rooms_per_person = tf.feature_column.bucketized_column(
        tf.feature_column.numeric_column("rooms_per_person"),
        boundaries=get_quantile_based_buckets(
            training_examples["rooms_per_person"], 10))

    long_x_lat = tf.feature_column.crossed_column(
        {bucketized_longitude, bucketized_latitude}, hash_bucket_size=1000)

    feature_columns = {long_x_lat, bucketized_longitude, bucketized_latitude, bucketized_housing_median_age,
                       bucketized_total_rooms, bucketized_total_bedrooms, bucketized_population, bucketized_households,
                       bucketized_median_income, bucketized_rooms_per_person}

    return feature_columns


###########################################################################################

def model_size(estimator):
    """
    Function calculates the size of a model.

    :param estimator: the model we are using.
    :return:  size of the model.
    """
    variables = estimator.get_variable_names()
    size = 0
    for variable in variables:
        if not any(x in variable
                   for x in ['global_step',
                             'centered_bias_weight',
                             'bias_weight',
                             'Ftrl']
                   ):
            size += np.count_nonzero(estimator.get_variable_value(variable))
    return size

###########################################################################################

def train_linear_classifier_model(
        learning_rate,
        regularization_strength,
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
      regularization_strength: A `float` that indicates the strength of the L1
         regularization. A value of `0.0` means no regularization.
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
      A `LinearClassifier` object trained on the training data.
    """

    periods = 7
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                          l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value_is_high"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value_is_high"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value_is_high"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on validation data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()
    plt.show()

    return linear_classifier

###########################################################################################

"""
Main function.  Execute the program.
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
    Task 1: Find a good regularization coefficient.
    
    Find an L1 regularization strength parameter which satisfies both constraints — 
    model size is less than 600 and log-loss is less than 0.35 on validation set.
    
    Baseline values:
    
        Training model...
    LogLoss (on validation data):
      period 00 : 0.31
      period 01 : 0.28
      period 02 : 0.27
      period 03 : 0.26
      period 04 : 0.26
      period 05 : 0.25
      period 06 : 0.25
    Model training finished.
    Model size: 789
    
    """

    # Train the model.

    start_time = time.time()

    linear_classifier = train_linear_classifier_model(
        learning_rate=0.1,
        # TWEAK THE REGULARIZATION VALUE BELOW
        regularization_strength=0.05,
        steps=300,
        batch_size=100,
        feature_columns=construct_feature_columns(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
    print("Model size:", model_size(linear_classifier))

    end_time = time.time()

    print("Time taken to train model: " + str(end_time - start_time))

###########################################################################################

"""
Exercise 9.2 Questions:
###########################################################################################

Why are we regularizing with respect to sparsity?

Sparse vector: A vector whose values are mostly zeroes.

Sparse feature: A feature vector whose values are predominantly zero or empty.

Sparse representation: A representation of a tensor that only stores nonzero element.

To make training the model less computationally expensive and faster.

Some feature vectors are high dimension and sparse, resulting in mostly "0" values and a few "1" values.  Using
these feature vectors to create feature crosses or other synthetic features increases the model size drastically.
If we can effectively remove them from consideration then we can reduce model size, which makes training the model 
less computationally expensive and faster.

This will also help to avoid overfitting and generally make the model more efficient.

########################################################

How does L1 regularization increase sparsity?

Unlike L2 regularization, L1 regularization forces weights to exactly 0.0.

L1 regularization penalizes | weight |, or the absolute value of weight.

L1 regularization thus has a derivative that is a constant, k, independent of weight.

L1 regularization subtracts some constant value, k, from the weight iteratively.  When the subtraction results crosses
 0.0, it is zeroed out. (due to the abs function having a discontinuity at 0)

If a high dimensional sparse feature vector can have its weight forced to exactly 0.0, it is essentially removed
from consideration when training the model.  Hence, "zero'ing" out the feature set will save on RAM requirements
and could reduce noise in the model.

########################################################

Task 1: Here, just report the best log loss value / model size you can get and what gamma value you used to get them.

Training model...
LogLoss (on validation data):
  period 00 : 0.31
  period 01 : 0.28
  period 02 : 0.26
  period 03 : 0.25
  period 04 : 0.25
  period 05 : 0.24
  period 06 : 0.24
Model training finished.
Model size: 784
Time taken to train model: 210.67125177383423

Process finished with exit code 0

Yea, I'm not going to do many attempts since it takes nearly 4 minutes per. (even longer using notebook online)

########################################################
Save your answers in lab09_2.txt.
"""
