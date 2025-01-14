"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 09 - Classification

Notes:

Exercise 9.1 - Logistic Regression

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
    # Create a boolean categorical feature representing whether the
    # median_house_value is above a set threshold.
    output_targets["median_house_value_is_high"] = (
            california_housing_dataframe["median_house_value"] > 265000).astype(float)
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

def train_linear_regressor_model(
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

def train_linear_classifier_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model.

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
      A `LinearClassifier` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear classifier object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    # YOUR CODE HERE: Construct the linear classifier.
    # URL: https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(training_examples),
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
    print("LogLoss (on training data):")
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

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
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

    # Train linear regression model.
    linear_regressor = train_linear_regressor_model(
        learning_rate=0.000001,
        steps=200,
        batch_size=20,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    # """
    # Task 1: Can We Calculate LogLoss for These Predictions?
    #
    # Given the predictions and the targets, can we calculate LogLoss?
    #
    # No, because the histogram shows show that we would be trying to take the log of negative values which is
    # outside the domain of the logarithmic function.
    #
    # i.e. log(1 - 3000) = log(-2999) = error
    #
    # So, we would have to normalize the results first.
    #
    # """
    #
    # # # Assignment for clarity.
    # # test_examples = validation_examples
    # # test_targets = validation_targets
    # #
    # # # Obtain predictions.
    # # predict_test_input_fn = lambda: my_input_fn(
    # #     test_examples,
    # #     test_targets["median_house_value_is_high"],
    # #     num_epochs=1,
    # #     shuffle=False)
    # #
    # # test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    # # test_predictions = np.array([item['predictions'][0] for item in test_predictions])
    # #
    # # root_mean_squared_error = math.sqrt(
    # #     metrics.mean_squared_error(test_predictions, test_targets))
    # #
    # # print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
    # #
    # # _ = plt.hist(test_predictions)
    # # plt.show()
    #
    # """
    # Task 2: Train a Logistic Regression Model and Calculate LogLoss on the Validation Set
    #
    # Code Added to "def train_linear_classifier_model(" function:
    #
    # # YOUR CODE HERE: Construct the linear classifier.
    # # URL: https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier
    # linear_classifier = tf.estimator.LinearClassifier(
    #     feature_columns=construct_feature_columns(training_examples),
    #     optimizer=my_optimizer
    # )
    # """
    #
    # # Train logistic regression model.
    # linear_classifier = train_linear_classifier_model(
    #     learning_rate=0.000005,
    #     steps=1000,
    #     batch_size=40,
    #     training_examples=training_examples,
    #     training_targets=training_targets,
    #     validation_examples=validation_examples,
    #     validation_targets=validation_targets)
    #
    # """
    # Task 3: Calculate Accuracy and plot a ROC Curve for the Validation Set
    #
    # Verify if all metrics improve at the same time.
    #
    # Yep, I managed to improve both at the same time.
    #
    # Google Baseline:
    #
    # AUC on the validation set: 0.70
    # Accuracy on the validation set: 0.75
    # """
    #
    # # Create input functions.
    # predict_training_input_fn = lambda: my_input_fn(training_examples,
    #                                                 training_targets["median_house_value_is_high"],
    #                                                 num_epochs=1,
    #                                                 shuffle=False)
    # predict_validation_input_fn = lambda: my_input_fn(validation_examples,
    #                                                   validation_targets["median_house_value_is_high"],
    #                                                   num_epochs=1,
    #                                                   shuffle=False)
    #
    # # Calculate AUC and Accuracy metrics.
    # evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
    #
    # print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
    # print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
    #
    # # Plot a ROC curve.
    # validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    # # Get just the probabilities for the positive class.
    # validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])
    #
    # false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    #     validation_targets, validation_probabilities)
    # plt.plot(false_positive_rate, true_positive_rate, label="our model")
    # plt.plot([0, 1], [0, 1], label="random classifier")
    # _ = plt.legend(loc=2)
    # plt.show()

###########################################################################################

"""
Exercise 9.1 Questions:
###########################################################################################

How does the linear regression approach to the problem fare?

Training model...
RMSE (on training data):
  period 00 : 0.45
  period 01 : 0.45
  period 02 : 0.45
  period 03 : 0.44
  period 04 : 0.44
  period 05 : 0.44
  period 06 : 0.44
  period 07 : 0.44
  period 08 : 0.44
  period 09 : 0.44
Model training finished.
Final RMSE (on test data): 0.44

Process finished with exit code 0

There seems to be spikes in RMSE values for some periods from lower to higher then back to lower values for both the
training and validation set.

It isn't a great fit and may be overfitting because validation RMSE is higher than the training RMSE by a decent
margin.

(refer to exercise9.1_task1_linear_regression_rmse.png screen capture in Lab09 directory)

########################################################

Task 1: Compare and contrast L2 Loss vs LogLoss.

L2 Loss:

Function calculates the squares of the difference between a model's predicted value for a labeled example and the
actual value of the label.

Amplifies the influence of bad predictions due to the squaring. (reacts strongly to outlier values)

Doesn't penalize false classification very well.

Want to minimize this value.

LogLoss:

Measures the performance of a model where the output are probability values between 0 and 1. (via sigmoid function)
(necessary in order to prevent taking log of negative values)

Probability values at exactly 0 or 1 are undefined.

Uses natural logarithm - base "e".

Penalizes false classification a lot better than L2 Loss.

Want to minimize this value.

########################################################

Task 2: Explain how the logistic regression fares compared to the linear regression.

Training model...
LogLoss (on training data):
  period 00 : 0.61
  period 01 : 0.58
  period 02 : 0.57
  period 03 : 0.55
  period 04 : 0.55
  period 05 : 0.54
  period 06 : 0.53
  period 07 : 0.53
  period 08 : 0.53
  period 09 : 0.53
Model training finished.

Process finished with exit code 0

Higher RMSE values but it more closely fits between the training and validation sets.  So, a pretty good fit since
validation error is only slightly higher than training error.

(refer to exercise9.1_task2_linear_classifier_results.png screen capture in Lab09 directory)

########################################################

Task 3: Here, just report the best values you can achieve for AUC/accuracy and 
what hyperparameters you used to get them.

    # Train logistic regression model.
    linear_classifier = train_linear_classifier_model(
        learning_rate=0.000005,
        steps=1000,
        batch_size=40,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)
        
        
Training model...
LogLoss (on training data):
  period 00 : 0.57
  period 01 : 0.55
  period 02 : 0.54
  period 03 : 0.53
  period 04 : 0.53
  period 05 : 0.53
  period 06 : 0.53
  period 07 : 0.54
  period 08 : 0.52
  period 09 : 0.53
Model training finished.
AUC on the validation set: 0.74
Accuracy on the validation set: 0.76

Process finished with exit code 0

########################################################

Other console output:

D:\Dropbox\cs344-ai\venv3.6-64bit\Scripts\python.exe D:/Dropbox/cs344-ai/cs344/Labs/Lab09/lab09_1.py
Training examples summary:
       latitude  longitude  ...  median_income  rooms_per_person
count   12000.0    12000.0  ...        12000.0           12000.0
mean       35.6     -119.6  ...            3.9               2.0
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
mean       35.6     -119.6  ...            3.9               2.0
std         2.1        2.0  ...            1.9               1.1
min        32.5     -124.3  ...            0.5               0.1
25%        33.9     -121.8  ...            2.6               1.5
50%        34.2     -118.5  ...            3.5               1.9
75%        37.7     -118.0  ...            4.7               2.3
max        41.9     -114.6  ...           15.0              41.3

[8 rows x 9 columns]
Training targets summary:
       median_house_value_is_high
count                     12000.0
mean                          0.3
std                           0.4
min                           0.0
25%                           0.0
50%                           0.0
75%                           1.0
max                           1.0
Validation targets summary:
       median_house_value_is_high
count                      5000.0
mean                          0.2
std                           0.4
min                           0.0
25%                           0.0
50%                           0.0
75%                           0.0
max                           1.0

Save your answers in lab09_1.txt.
"""
