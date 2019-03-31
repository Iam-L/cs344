"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-31-19
Assignment: Lab 10 - Neural Networks

Notes:

Exercise 10.2 - Improving Neural Net Performance

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
    """Trains a neural network model.

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

def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.

    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNRegressor` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
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
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
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

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor, training_rmse, validation_rmse


###########################################################################################

def linear_scale(series):
    """
    Function normalize the inputs to fall within the range -1, 1.

    :param series: un-normalized values.
    :return: normalized values.
    """

    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


###########################################################################################

def normalize_linear_scale(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
    #
    # Your code here: normalize the inputs.
    #
    normalized_dataframe = examples_dataframe

    normalized_dataframe["latitude"] = linear_scale(examples_dataframe["latitude"])
    normalized_dataframe["longitude"] = linear_scale(examples_dataframe["longitude"])
    normalized_dataframe["housing_median_age"] = linear_scale(examples_dataframe["housing_median_age"])
    normalized_dataframe["total_rooms"] = linear_scale(examples_dataframe["total_rooms"])
    normalized_dataframe["total_bedrooms"] = linear_scale(examples_dataframe["total_bedrooms"])
    normalized_dataframe["population"] = linear_scale(examples_dataframe["population"])
    normalized_dataframe["households"] = linear_scale(examples_dataframe["households"])
    normalized_dataframe["median_income"] = linear_scale(examples_dataframe["median_income"])
    normalized_dataframe["rooms_per_person"] = linear_scale(examples_dataframe["rooms_per_person"])

    return normalized_dataframe


###########################################################################################

def log_normalize(series):
    """
    Function log-scales feature values.

    :param series:
    :return:
    """
    return series.apply(lambda x: math.log(x + 1.0))


def clip(series, clip_to_min, clip_to_max):
    """
    Function clips to specified min, max allowed  feature values.

    :param series:
    :param clip_to_min:
    :param clip_to_max:
    :return:
    """
    return series.apply(lambda x: (
        min(max(x, clip_to_min), clip_to_max)))


def z_score_normalize(series):
    """
    Function calculates z_scores for all feature values.

    :param series:
    :return:
    """
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    """
    Function assigns binary values of 0,1 based on threshold to all feature values.

    :param series:
    :param threshold:
    :return:
    """
    return series.apply(lambda x: (1 if x > threshold else 0))


###########################################################################################

def normalize(examples_dataframe):
    """Returns a version of the input `DataFrame` that has all its features normalized."""
    #
    # YOUR CODE HERE: Normalize the inputs.
    #
    normalized_dataframe = examples_dataframe

    normalized_dataframe["latitude"] = linear_scale(examples_dataframe["latitude"])
    normalized_dataframe["longitude"] = linear_scale(examples_dataframe["longitude"])
    normalized_dataframe["housing_median_age"] = clip(examples_dataframe["housing_median_age"], 1, 80)
    normalized_dataframe["total_rooms"] = z_score_normalize(examples_dataframe["total_rooms"])
    normalized_dataframe["total_bedrooms"] = binary_threshold(examples_dataframe["total_bedrooms"], 2)
    normalized_dataframe["population"] = log_normalize(examples_dataframe["population"])
    normalized_dataframe["households"] = z_score_normalize(examples_dataframe["households"])
    normalized_dataframe["median_income"] = log_normalize(examples_dataframe["median_income"])
    normalized_dataframe["rooms_per_person"] = z_score_normalize(examples_dataframe["rooms_per_person"])

    return normalized_dataframe


###########################################################################################

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

# # Train the model.
# _ = train_nn_regression_model(
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

"""
Task 1: Normalize the Features Using Linear Scaling
"""

# Normalize the feature set examples.
normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

# # Train the model on normalized data.
# _ = train_nn_regression_model(
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

"""
Task 2: Task 2: Try a Different Optimizer
"""

# # Train the model on normalized data and using the AdagradOptimizer.
# _ = train_nn_regression_model(
#     my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.007),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

# # Train the model on normalized data and using the AdamOptimizer.
# _ = train_nn_regression_model(
#     my_optimizer=tf.train.AdamOptimizer(learning_rate=0.0007),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

"""
Task 3: Explore Alternate Normalization Methods
"""

# Histogram of all features.
_ = normalized_training_examples.hist(bins=20, figsize=(18, 12), xlabelsize=10)
plt.show()

# Normalize feature values.
normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

# Train the model.
_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

###########################################################################################

"""
Exercise 10.2 Questions:
###########################################################################################

What does AdaGrad do to boost performance?

According to "improving_neural_net_performance.ipynb":

"The key insight of Adagrad is that it modifies the learning rate adaptively for each coefficient in a model, 
monotonically lowering the effective learning rate. This works great for convex problems, but isn't always ideal 
for the non-convex problem Neural Net training."

##################################################

Tasks 1–4: Report your best hyperparameter settings and their resulting performance.

Note: There is no task 4.

##########################
Task 1:

# Train the model on normalized data.
_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
    
Training model...
RMSE (on training data):
  period 00 : 236.19
  period 01 : 224.60
  period 02 : 192.59
  period 03 : 140.61
  period 04 : 120.12
  period 05 : 116.35
  period 06 : 112.43
  period 07 : 107.92
  period 08 : 102.44
  period 09 : 96.05
Model training finished.
Final RMSE (on training data):   96.05
Final RMSE (on validation data): 96.44

Process finished with exit code 0

##########################
Task 2:

# Train the model on normalized data and using the AdagradOptimizer.
_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
    
Training model...
RMSE (on training data):
  period 00 : 219.65
  period 01 : 199.51
  period 02 : 179.21
  period 03 : 159.55
  period 04 : 141.83
  period 05 : 127.84
  period 06 : 119.79
  period 07 : 117.20
  period 08 : 115.92
  period 09 : 114.91
Model training finished.
Final RMSE (on training data):   114.91
Final RMSE (on validation data): 114.71

Process finished with exit code 0

# Train the model on normalized data and using the AdamOptimizer.
_ = train_nn_regression_model(
    my_optimizer=tf.train.AdamOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
    
Training model...
RMSE (on training data):
  period 00 : 214.34
  period 01 : 133.66
  period 02 : 114.16
  period 03 : 107.53
  period 04 : 97.91
  period 05 : 85.99
  period 06 : 76.38
  period 07 : 72.80
  period 08 : 71.57
  period 09 : 70.88
Model training finished.
Final RMSE (on training data):   70.88
Final RMSE (on validation data): 71.04

Process finished with exit code 0

##########################
Task 3:

# Normalize feature values.
normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

# Train the model.
_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)
    
Training model...
RMSE (on training data):
  period 00 : 175.61
  period 01 : 134.03
  period 02 : 129.38
  period 03 : 123.92
  period 04 : 117.95
  period 05 : 114.07
  period 06 : 112.59
  period 07 : 111.45
  period 08 : 110.52
  period 09 : 109.64
Model training finished.
Final RMSE (on training data):   109.64
Final RMSE (on validation data): 107.84

Process finished with exit code 0

##################################################

Optional Challenge: You can skip this exercise.

^_^

##################################################
"""
