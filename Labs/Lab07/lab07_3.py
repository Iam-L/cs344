"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-13-19
Assignment: Lab 07 - Regression

Notes:

Exercise 7.3 - Synthetic Features and Outliers

-Included code just to see if I can run it from within PyCharm.

Resources:

URL: https://databricks.com/tensorflow/using-a-gpu
URL: https://dzone.com/articles/how-to-train-tensorflow-models-using-gpus
https://www.tensorflow.org/guide/using_gpu

Well, looks like my older laptap with Geforce 780M in SLI only has CUDA compute 3.0 so I can't use tensorflow with it.
However, my newer laptop with a Geforce 1050Ti has CUDA computer 6.1 so I'm good using that =).

URL: https://matplotlib.org/tutorials/introductory/usage.html
URL: https://matplotlib.org/users/pyplot_tutorial.html

"""

###########################################################################################

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

###########################################################################################

"""
First, we'll import the California housing data into a pandas DataFrame:
"""
california_housing_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe)

###########################################################################################

"""
Next, we'll set up our input function, and define the function for model training:
"""


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

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
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


###########################################################################################

"""
Next, we'll set up our input function, and define the function for model training:
"""


def train_model(learning_rate, steps, batch_size, input_feature):
    """Trains a linear regression model.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.

    Returns:
      A Pandas `DataFrame` containing targets and the corresponding predictions done
      after training the model.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature

    my_feature_data = california_housing_dataframe[[my_feature]].astype('float32')
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label].astype('float32')

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    # Set up to plot the state of our model's line each period.
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Learned Line by Period")
    plt.ylabel(my_label)
    plt.xlabel(my_feature)
    sample = california_housing_dataframe.sample(n=300)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period,
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        # Compute loss.
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        root_mean_squared_errors.append(root_mean_squared_error)
        # Finally, track the weights and biases over time.
        # Apply some math to ensure that the data and line are plotted neatly.
        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)

    # Create a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

    """
    Code for Task 2.
    ####################################################
    """

    # Scatter-plot graph of Predictions versus Targets.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylabel('Target')
    plt.xlabel('Predictions')
    plt.title("Predictions versus Target Values")
    plt.tight_layout()
    plt.scatter(calibration_data["predictions"], calibration_data["targets"])

    # Histogram graph of rooms_per_person
    plt.subplot(1, 2, 2)
    plt.ylabel('Population')
    plt.xlabel('Rooms')
    plt.title("Rooms per Person")
    plt.tight_layout()
    plt.hist(california_housing_dataframe["rooms_per_person"])
    plt.show()

    """
    ####################################################
    """

    return calibration_data


###########################################################################################

if __name__ == '__main__':
    """
    Main function.  Executes training.
    """
    # Creates a graph.
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))

    ###############################################

    """
    Task 1: Try a Synthetic Feature
    
    Both the total_rooms and population features count totals for a given city block.

    But what if one city block were more densely populated than another? 
    We can explore how block density relates to median house value by creating a synthetic feature 
    that's a ratio of total_rooms and population.
    
    In the cell below, create a feature called rooms_per_person, and use that as the input_feature to train_model().

    What's the best performance you can get with this single feature by tweaking the learning rate? 
    (The better the performance, the better your regression line should fit the data, 
    and the lower the final RMSE should be.)
    
    """

    california_housing_dataframe["rooms_per_person"] = california_housing_dataframe['total_rooms'] / \
                                                       california_housing_dataframe['population']

    print(california_housing_dataframe[['rooms_per_person']])

    """
    Code for Task 3.
    ####################################################
    """

    # Clip so that we have no values over 1000.
    california_housing_dataframe["clipped_rooms_per_person"] = \
        california_housing_dataframe["rooms_per_person"].apply(lambda clip: min(clip, 7))

    """
    ####################################################
    """

    calibration_data = train_model(
        learning_rate=0.0000005,
        steps=250,
        batch_size=5,
        input_feature="clipped_rooms_per_person"
    )

    ###############################################

    """
    Task 2: Identify Outliers
    
    We can visualize the performance of our model by creating a scatter plot of predictions vs. target values. 
    Ideally, these would lie on a perfectly correlated diagonal line.

    Use Pyplot's scatter() to create a scatter plot of predictions vs. targets, 
    using the rooms-per-person model you trained in Task 1.
    
    Do you see any oddities? Trace these back to the source data by looking at the distribution of
     values in rooms_per_person.
     
     Notes:
     
     Refer to code in the def train_model function in the section labeled "Code for Task 2". (or refer below)
     Refer to .png files in Lab07 directory for the scatter-plot and histogram graphs.
     
    # Scatter-plot graph of Predictions versus Targets.
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylabel('Target')
    plt.xlabel('Predictions')
    plt.title("Predictions versus Target Values")
    plt.tight_layout()
    plt.scatter(calibration_data["predictions"], calibration_data["targets"])

    # Histogram graph of rooms_per_person
    plt.subplot(1, 2, 2)
    plt.ylabel('Population')
    plt.xlabel('Rooms')
    plt.title("Rooms per Person")
    plt.tight_layout()
    plt.hist(california_housing_dataframe["rooms_per_person"])
    plt.show()
    
    """

    ###############################################

    """
    Task 3: Clip Outliers
    
    See if you can further improve the model fit by setting the outlier values of rooms_per_person 
    to some reasonable minimum or maximum.
    
    For reference, here's a quick example of how to apply a function to a Pandas Series:
    
    clipped_feature = my_dataframe["my_feature_name"].apply(lambda x: max(x, 0))
    The above clipped_feature will have no values less than 0.
    
    Notes:
    
     Refer to code in the def train_model function in the section labeled "Code for Task 3". (or refer below)
     Refer to .png files in Lab07 directory for the scatter-plot and histogram graphs.
     
     # Clip so that we have no values over 1000.
    california_housing_dataframe["clipped_rooms_per_person"] = \
        california_housing_dataframe["rooms_per_person"].apply(lambda clip: min(clip, 7))

    """
###########################################################################################
###########################################################################################

"""
Exercise 7.3 Questions:
###########################################################################################
What is the purpose of introducing synthetic features?
#######################################################

A feature not present among the input features, but created from one or more of them.

The purpose is to improve upon the performance and predictive abilities of the machine learning model.

URL: http://www.chioka.in/improving-classifier-performance-with-synthetic-features/

###########################################################################################
What are outliers and what is typically done with them?
#######################################################

Values distant from most other values. In machine learning, any of the following are outliers:

Weights with high absolute values.
Predicted values relatively far away from the actual values.
Input data whose values are more than roughly 3 standard deviations from the mean.

#######################################################

Outliers often cause problems in model training. Clipping is one way of managing outliers.

###########################################################################################

Submit solutions to tasks 1–3.

"""

###########################################################################################
###########################################################################################

