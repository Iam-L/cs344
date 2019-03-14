"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-13-19
Assignment: Lab 07 - Regression

Notes:

Exercise 7.2 - First Steps with Tensor Flow

-Included code just to see if I can run it from within PyCharm.

Resources:

URL: https://databricks.com/tensorflow/using-a-gpu
URL: https://dzone.com/articles/how-to-train-tensorflow-models-using-gpus
https://www.tensorflow.org/guide/using_gpu

Well, looks like my older laptap with Geforce 780M in SLI only has CUDA compute 3.0 so I can't use tensorflow with it.
However, my newer laptop with a Geforce 1050Ti has CUDA computer 6.1 so I'm good using that =).

"""

###########################################################################################

import math
import time

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

###########################################################################################

# Load data set.
california_housing_dataframe = \
    pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

"""
We'll randomize the data, just to be sure not to get any pathological ordering effects that might harm 
the performance of Stochastic Gradient Descent. Additionally, we'll scale median_house_value to be in 
units of thousands, so it can be learned a little more easily with learning rates in a range that we usually use.
"""
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe)

"""
We'll print out a quick summary of a few useful statistics on each column: 
count of examples, mean, standard deviation, max, min, and various quantiles.
"""
print(california_housing_dataframe.describe())

###########################################################################################

"""
Step 1: Define Features and Configure Feature Columns

In order to import our training data into TensorFlow, we need to specify what type of data each feature contains. 
There are two main types of data we'll use in this and future exercises:

Categorical Data: Data that is textual. In this exercise, our housing data set does not contain 
any categorical features, but examples you might see would be the home style, the words in a real-estate ad.

Numerical Data: Data that is a number (integer or float) and that you want to treat as a number. 
As we will discuss more later sometimes you might want to treat numerical data (e.g., a postal code) 
as if it were categorical.

In TensorFlow, we indicate a feature's data type using a construct called a feature column. 
Feature columns store only a description of the feature data; they do not contain the feature data itself.

To start, we're going to use just one numeric input feature, total_rooms. 
The following code pulls the total_rooms data from our california_housing_dataframe and 
defines the feature column using numeric_column, which specifies its data is numeric:
"""

# Define the input feature: total_rooms.
# my_feature = california_housing_dataframe[["total_rooms"]]

# Define the input feature: population.
my_feature = california_housing_dataframe[["population"]]

# Configure a numeric feature column for total_rooms.
# feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Configure a numeric feature column for population.
feature_columns = [tf.feature_column.numeric_column("population")]

"""
Step 2: Define the Target

Next, we'll define our target, which is median_house_value. 
Again, we can pull it from our california_housing_dataframe:
"""

# Define the label.
targets = california_housing_dataframe["median_house_value"]

###########################################################################################

"""
Step 3: Configure the LinearRegressor

Next, we'll configure a linear regression model using LinearRegressor. 
We'll train this model using the GradientDescentOptimizer, 
which implements Mini-Batch Stochastic Gradient Descent (SGD). 
The learning_rate argument controls the size of the gradient step.

NOTE: To be safe, we also apply gradient clipping to our optimizer via clip_gradients_by_norm. 
Gradient clipping ensures the magnitude of the gradients do not become too large during training, 
which can cause gradient descent to fail.
"""

# Use gradient descent as the optimizer for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)

###########################################################################################

"""
Step 4: Define the Input Function

To import our California housing data into our LinearRegressor, we need to define an input function, 
which instructs TensorFlow how to preprocess the data, as well as how to batch, shuffle, 
and repeat it during model training.

First, we'll convert our pandas feature data into a dict of NumPy arrays. 
We can then use the TensorFlow Dataset API to construct a dataset object from our data, and then break our data 
into batches of batch_size, to be repeated for the specified number of epochs (num_epochs).

NOTE: When the default value of num_epochs=None is passed to repeat(), the input data will be repeated indefinitely.

Next, if shuffle is set to True, we'll shuffle the data so that it's passed to the model randomly during training. 
The buffer_size argument specifies the size of the dataset from which shuffle will randomly sample.

Finally, our input function constructs an iterator for the dataset and 
returns the next batch of data to the LinearRegressor.
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
Step 5: Train the Model

We can now call train() on our linear_regressor to train the model. 
We'll wrap my_input_fn in a lambda so we can pass in my_feature and targets as arguments 
(see this TensorFlow input function tutorial for more details), and to start, we'll train for 100 steps.

Step 6: Evaluate the Model

Let's make predictions on that training data, to see how well our model fit it during training.

NOTE: Training error measures how well your model fits the training data, 
but it does not measure how well your model generalizes to new data. 
In later exercises, you'll explore how to split your data to evaluate your model's ability to generalize.
"""


def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    """Trains a linear regression model of one feature.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      input_feature: A `string` specifying a column from `california_housing_dataframe`
        to use as input feature.
    """

    periods = 10
    steps_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    # Create input functions.
    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, num_epochs=1, shuffle=False)

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
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
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
    plt.show()

    # Output a table with calibration data.
    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)


###########################################################################################

if __name__ == '__main__':
    """
    Main function.
    Executes training.
    """
    start_time = time.time()

    train_model(
        learning_rate=0.0001,
        steps=300,
        batch_size=300
    )

    end_time = time.time()

    print("Time taken to train: " + str((end_time - start_time)))

###########################################################################################
###########################################################################################

"""
Exercise 7.2 Questions:
###########################################################################################
Compare and contrast categorical vs numerical data

Categorical data:

Data that is textual.

###################################################

Numerical data:

Data that is a number (integer or float) and that you want to treat as a number.

###########################################################################################

Submit solutions to tasks 1–2. Include your best hyper-parameter values and the resulting RMSE, 
but not the training output.

Task 1:

        learning_rate=0.0001,
        steps=300,
        batch_size=100
        
Final RMSE (on training data): 166.53
Time taken to train: 64.14380383491516

Takes 64.1 seconds per attempt, so I'm going to go with this one as it beat the target RMSE of 180.

###################################################

Task 2:

Changed the following lines to implement a different feature.

# Define the input feature: total_rooms.
# my_feature = california_housing_dataframe[["total_rooms"]]

# Define the input feature: population.
my_feature = california_housing_dataframe[["population"]]

# Configure a numeric feature column for total_rooms.
# feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Configure a numeric feature column for population.
feature_columns = [tf.feature_column.numeric_column("population")]

###################################################

        learning_rate=0.0001,
        steps=300,
        batch_size=100
        
Final RMSE (on training data): 166.39
Time taken to train: 67.13072752952576

Well, it did go down a bit using the same settings. (I made it worse with other attempts =( )

###########################################################################################
What are the hyper-parameters learned in these exercises and is there a “standard” tuning algorithm for them?

The "knobs" that you tweak during successive runs of training a model.

We are tweaking the following parameters:

learning rate
step size
batch size

###################################################

There are no standard tuning algorithm as it depends on what your data consists of.

"The short answer is that the effects of different hyperparameters are data dependent. 
So there are no hard-and-fast rules; you'll need to test on your data."

###########################################################################################

"""
