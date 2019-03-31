"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 08 - Feature Engineering

Notes:

Exercise 8.3 - Regression: Predict Fuel Efficiency

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.
"""

###########################################################################################

from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
print()

###########################################################################################

# Download dataset.
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)
print()

# Convert to Pandas dataframe.
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

###########################################################################################

# Make a copy of the dataset.
dataset = raw_dataset.copy()
print(dataset.tail())
print()

# Display features.
print(dataset.isna().sum())
print()

# Drop unwanted features.
dataset = dataset.dropna()
print(dataset.tail())
print()

# Get just the "origin" feature.
origin = dataset.pop('Origin')

# Convert "origin" feature to one-hot binary.
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())
print()

# Split into training and testing set.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Visualize the dataset.
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# Display overall statistics.
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print(train_stats)
print()

# Split into target values (labels)
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

###########################################################################################

"""
Note: Although we intentionally generate these statistics from only the training dataset, these statistics will also 
be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution 
that the model has been trained on.
"""


def norm(x):
    """
    Function normalizes the input data.

    :param x: input data.
    :return:  normalized data.
    """
    return (x - train_stats['mean']) / train_stats['std']


###########################################################################################

"""
Caution: The statistics used to normalize the inputs here (mean and standard deviation) need to be applied to any 
other data that is fed to the model, along with the one-hot encoding that we did earlier. 
That includes the test set as well as live data when the model is used in production.
"""

# Normalize the data.
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


###########################################################################################

def build_model():
    """
    Functions builds a Sequential training model.

    :return: training model.
    """
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


###########################################################################################

# Build the model.
model = build_model()

# Inspect the model.
model.summary()

# Test the model with some examples.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


###########################################################################################

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


###########################################################################################

# Length of training.
EPOCHS = 1000

# Train the model.
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

###########################################################################################

# Visualize training progress.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


###########################################################################################

def plot_history(history):
    """
    Visualize training results using graphs.

    :param history: training results.
    :return: nothing.
    """
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


###########################################################################################

# Display training results.
plot_history(history)
plt.show()

###########################################################################################
"""
Let's update the model.fit call to automatically stop training when the validation score doesn't improve. 
We'll use an EarlyStopping callback that tests a training condition for every epoch. 
If a set amount of epochs elapses without showing improvement, then automatically stop the training.
"""
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()

###########################################################################################

# Test how well the model generalizes using test dataset.
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

###########################################################################################

# Make predictions using test datset.
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Display error distribution for predictions.
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

"""
Mean Squared Error (MSE) is a common loss function used for regression problems 
(different loss functions are used for classification problems).

Similarly, evaluation metrics used for regression differ from classification. 
A common regression metric is Mean Absolute Error (MAE).

When numeric input data features have values with different ranges, 
each feature should be scaled independently to the same range.

If there is not much training data, one technique is to prefer a small network with 
few hidden layers to avoid overfitting.

Early stopping is a useful technique to prevent overfitting.
"""

###########################################################################################

"""
Exercise 8.3 Questions:
###########################################################################################
Compare and contrast Seaborn vs. MatPlotLib.

Seaborn builds on top of Matplotlib and introduces additional plot types. 
It also makes your traditional Matplotlib plots look a bit prettier.

You should be using both at the same time.

##############################################

Matplotlib and seaborn are the two important library for data visualisation.

Seaborn library is built on top of the Matplotlib library.

You can learn both the libraries as both them are easy to understand and implement. 
If you are a beginner and want to learn visualisation fast then I would recommend you to learn seaborn first.

Plotting with seaborn is very much easy and intuitive and you will learn it very fast. 
Once you are comfortable with the seaborn then you can start with matplotlib.

##############################################

URL: https://www.quora.com/What-is-the-difference-between-Matplotlib-and-Seaborn-Which-one-should-I-learn-for-studying-data-science

########################################################
How big is this dataset and does it seem of an appropriate size for this problem?

There are 397 examples in the dataset with 8 feature columns.

It seems as bit small to encompass all the different models of late 1970's and early 1980's automobiles.
Then again, I have no idea how many different models were in existence at the time globally.

########################################################
Explain what the prescribed normalization of the data does.

Increase training speed and ensure that it converges, which might not happen without normalization.
Also makes it independent on choice of units used in the input.

########################################################
Is this an example of a linear regression model?

Yes, based on the scatterplot of predictions versus true values there is a very high linear
correlation. (refer to exercise8.3_prediction_results.png)

########################################################

In their conclusion, they claim that smaller datasets “prefer” smaller networks. Do you agree? Explain your answer.

Yes, larger networks may lead to worse overfitting to the training dataset.

###########################################################################################

Submit your solutions to exercises 1–2.
"""
