"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-15-19

Homework 3 - Regression
Boston Housing Dataset - Load and Manipulate via Tensorflow

Notes:

############################################

Resources Used:

URL: https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb
(intro to Pandas)

URL: https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/video-lecture
(machine-learning crash course)

URL: https://matplotlib.org/tutorials/introductory/usage.html
(intro to matplotlib)

URL: https://docs.scipy.org/doc/numpy/user/quickstart.html
(numpy tutorial)

URL: https://www.tensorflow.org/guide/keras
(tensorflow - guide to keras)

URL: https://www.tensorflow.org/guide/using_gpu
(tensorflow - using CUDA)

URL: https://keras.io/datasets/#boston-housing-price-regression-dataset
(boston housing data-set)

URL: https://github.com/kvlinden-courses/cs344-code/blob/master/u06learning/numpy.ipynb
(numpy and karas)

URL: https://www.kaggle.com/shanekonaung/boston-housing-price-dataset-with-keras
(using keras)

URL: https://www.kaggle.com/c/boston-housing
(to find out what each attribute is for each column of the dataset)

URL: https://stackoverflow.com/questions/28503445/assigning-column-names-to-a-pandas-series
(assign column names to pandas series)

URL: https://stackoverflow.com/questions/11346283/renaming-columns-in-pandas
URL: https://chrisalbon.com/python/data_wrangling/pandas_rename_multiple_columns/
(rename columns)

############################################

Assignment Instructions:

Use Python/NumPy/Pandas/Keras to load and manipulate the Boston Housing Dataset as follows.

Compute the dimensions of the data structures. Include code to print these values.

Construct a suitable testing set, training set, and validation set for this data.
Include code to create these datasets but do not include the datasets themselves.

Create one new synthetic feature that could be useful for machine learning in this domain.
Explain what it is and why it might be useful.

Assignment Question:

Consider the task of constructing a network of perceptrons that computes the XOR function. If this is possible,
specify the network. If it is not possible, explain why it can’t be done.

"""
############################################################################################

# Import required libraries.
import math
import matplotlib.pyplot as matplotlib
import numpy as numpi
import pandas as pandas
import sklearn.metrics as metrics
import tensorflow as tensorflow
from tensorflow.python.data import Dataset

# Import dataset.
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# Set Global pandas and tensorflow options.
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
pandas.options.display.max_rows = 10
pandas.options.display.float_format = '{:.1f}'.format

# Activates or deactivates debug mode.
debug = 1

############################################################################################

# Rename variables for clarity.
boston_housing_keras_training_data = train_data
boston_housing_keras_training_targets = train_targets
boston_housing_keras_testing_data = test_data
boston_housing_keras_testing_targets = test_targets

############################################################################################
# Names associated with each column.
column_names = ['Crime_Rate', 'Zoning', 'Indus', 'Chas',
                'Nitrogen_Oxides', 'Avg_Rooms', 'Age', 'Distances',
                'Radial', 'Tax', 'Pupil_Teacher', 'Black', 'Low_status']

# Create Pandas dataframes for boston housing dataset.
boston_housing_pandas_dataframe_training_data = pandas.DataFrame(boston_housing_keras_training_data)
boston_housing_pandas_dataframe_training_targets = pandas.DataFrame(boston_housing_keras_training_targets)
boston_housing_pandas_dataframe_testing_data = pandas.DataFrame(boston_housing_keras_testing_data)
boston_housing_pandas_dataframe_testing_targets = pandas.DataFrame(boston_housing_keras_testing_targets)

# boston_housing_pandas_dataframe_training_data.rename(columns={'0': 'Crime Rate', '1': 'Zoning'}, inplace=True)
# boston_housing_pandas_dataframe_training_data.rename({'0': 'Crime Rate', '1': 'Zoning'}, axis=1)
# boston_housing_pandas_dataframe_training_data.set_axis(
#     ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm'], axis='columns', inplace=False)

# Rename columns in Pandas dataframes for boston housing dataset.
boston_housing_pandas_dataframe_training_data.columns = column_names
boston_housing_pandas_dataframe_training_targets.columns = ["Median_Housing_Value"]
boston_housing_pandas_dataframe_testing_data.columns = column_names
boston_housing_pandas_dataframe_testing_targets.columns = ["Median_Housing_Value"]

if debug:
    print("boston_housing_pandas_dataframe_training_data total columns:")
    print(boston_housing_pandas_dataframe_training_data.head())
    print("")
    print("boston_housing_pandas_dataframe_training_targets total columns:")
    print(boston_housing_pandas_dataframe_training_targets.head())
    print("")

############################################################################################

# Shuffle the data randomly.
boston_housing_pandas_dataframe_training_data = boston_housing_pandas_dataframe_training_data.reindex(
    numpi.random.permutation(boston_housing_pandas_dataframe_training_data.index))
boston_housing_pandas_dataframe_training_targets = boston_housing_pandas_dataframe_training_targets.reindex(
    numpi.random.permutation(boston_housing_pandas_dataframe_training_targets.index))
boston_housing_pandas_dataframe_testing_data = boston_housing_pandas_dataframe_testing_data.reindex(
    numpi.random.permutation(boston_housing_pandas_dataframe_testing_data.index))
boston_housing_pandas_dataframe_testing_targets = boston_housing_pandas_dataframe_testing_targets.reindex(
    numpi.random.permutation(boston_housing_pandas_dataframe_testing_targets.index))


############################################################################################

def boston_housing_dataset_shape_dimensions_samples():
    """
    Function displays shape (dimension) and sample of boston housing dataset.
    """
    if debug:
        print('Keras - Boston Housing Dataset shape (dimensions) and samples')
        print('Training data dimensions : ' + str(boston_housing_keras_training_data.shape))
        print('Test data dimensions: ' + str(boston_housing_keras_testing_data.shape))
        print('Training targets dimensions: ' + str(boston_housing_keras_training_targets.shape))
        print('Test targets dimensions: ' + str(boston_housing_keras_testing_targets.shape))
        print('\n\n')
        print('Training sample:\n' + str(boston_housing_keras_training_data[0]))
        print('Training target sample:\n' + str(boston_housing_keras_training_targets[0]))
        print('\n\n')
        print('Test sample:\n' + str(boston_housing_keras_testing_data[0]))
        print('Test target sample:\n' + str(boston_housing_keras_testing_targets[0]))
        print('\n\n')


############################################################################################

def boston_housing_dataset_pandas_dataframe_dimensions_samples():
    if debug:
        print("Boston Housing Dataset - Pandas Dataframe Information.")
        print('Training data dimensions : ' + str(boston_housing_pandas_dataframe_training_data.shape))
        print('Test data dimensions: ' + str(boston_housing_pandas_dataframe_testing_data.shape))
        print('Training targets dimensions: ' + str(boston_housing_pandas_dataframe_training_targets.shape))
        print('Test targets dimensions: ' + str(boston_housing_pandas_dataframe_testing_targets.shape))
        print('\n\n')
        print('Training sample:\n' + str(boston_housing_pandas_dataframe_training_data))
        print("")
        print('Training target sample:\n' + str(boston_housing_pandas_dataframe_training_targets))
        print('\n\n')
        print('Test sample:\n' + str(boston_housing_pandas_dataframe_testing_data))
        print("")
        print('Test target sample:\n' + str(boston_housing_pandas_dataframe_testing_targets))
        print('\n\n')


############################################################################################

def preprocess_features(boston_housing_dataframe):
    """Prepares input features from Boston housing data set.

  Args:
    boston_housing_dataframe: A Pandas DataFrame expected to contain data
      from the Boston housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
    selected_features = boston_housing_dataframe[column_names]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["property_tax_per_capita_crime"] = (
            boston_housing_dataframe["Tax"] /
            boston_housing_dataframe["Crime_Rate"])

    """
    Synthetic Feature:
    
    full-value property-tax rate per $10,000 / per capita crime rate by town = property_tax_per_capita_crime
    
    I believe this should be a useful synthetic feature as property tax rate and per capita crime rate could have a high
    chance of influencing the median housing value in that area of boston.
    
    I feel that the median housing value in the area would decrease if there is a higher per capita crime rate.
    I feel that the median housing value in the area would increase if there is a higher property tax rate.
    """
    return processed_features


############################################################################################

def preprocess_targets(boston_housing_dataframe):
    """Prepares target features (i.e., labels) from Boston housing data set.

  Args:
    boston_housing_dataframe: A Pandas DataFrame expected to contain data
      from the Boston housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
    output_targets = pandas.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["Median_Housing_Value"] = (
        boston_housing_dataframe["Median_Housing_Value"])
    return output_targets


############################################################################################

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

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
    features = {key: numpi.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


############################################################################################

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
    return set([tensorflow.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


############################################################################################

def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      learning_rate: A `float`, the learning rate.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing one or more columns from
        `boston_housing_dataframe` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `boston_housing_dataframe` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `boston_housing_dataframe` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `boston_housing_dataframe` to use as target for validation.

    Returns:
      A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tensorflow.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tensorflow.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 1. Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, training_targets["Median_Housing_Value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["Median_Housing_Value"],
                                                    num_epochs=1, shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["Median_Housing_Value"],
                                                      num_epochs=1, shuffle=False)

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

        # 2. Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = numpi.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = numpi.array([item['predictions'][0] for item in validation_predictions])

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
    matplotlib.subplot(1, 1, 1)
    matplotlib.ylabel("RMSE")
    matplotlib.xlabel("Periods")
    matplotlib.title("Root Mean Squared Error vs. Periods")
    matplotlib.tight_layout()
    matplotlib.plot(training_rmse, label="training")
    matplotlib.plot(validation_rmse, label="validation")
    matplotlib.legend()
    matplotlib.show()

    return linear_regressor


############################################################################################
if __name__ == '__main__':
    """
    Executes the program.
    
    Note: Target column name is : medv - median value of owner-occupied homes in $1000's.
    """
    print("Boston Housing Dataset")

    # Print information concerning dataset.
    boston_housing_dataset_shape_dimensions_samples()

    # Print information concerning Pandas dataframes.
    boston_housing_dataset_pandas_dataframe_dimensions_samples()

    """
    For the training set, we'll choose the first 300 examples, out of the total of 404.
    """
    training_examples = preprocess_features(boston_housing_pandas_dataframe_training_data.head(300))
    print("\nTraining set examples:\n")
    print(training_examples.describe())

    print("\nTraining set targets:\n")
    training_targets = preprocess_targets(boston_housing_pandas_dataframe_training_targets.head(300))
    print(training_targets.describe())

    """
    For the validation set, we'll choose the last 104 examples, out of the total of 404.
    """
    validation_examples = preprocess_features(boston_housing_pandas_dataframe_training_data.tail(100))
    print("\nValidation set examples:\n")
    print(validation_examples.describe())

    validation_targets = preprocess_targets(boston_housing_pandas_dataframe_training_targets.tail(100))
    print("\nValidation set targets:\n")
    print(validation_targets.describe())

    """
    For the testing set, we'll choose the first 52 examples, out of the total of 102.
    """
    test_examples = preprocess_features(boston_housing_pandas_dataframe_testing_data.head(52))
    print("\nTesting set examples:\n")
    print(test_examples.describe())

    test_targets = preprocess_targets(boston_housing_pandas_dataframe_testing_targets.head(52))
    print("\nTesting set targets:\n")
    print(test_targets.describe())

    """
    Adjust parameters and train the model.
    (tweak to improve root mean squared error)
    """
    linear_regressor = train_model(
        learning_rate=0.00001,
        steps=500,
        batch_size=5,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets)

    """
    Compare against our testing data and targets.
    """
    predict_test_input_fn = lambda: my_input_fn(
        test_examples,
        test_targets["Median_Housing_Value"],
        num_epochs=1,
        shuffle=False)

    test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
    test_predictions = numpi.array([item['predictions'][0] for item in test_predictions])

    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(test_predictions, test_targets))

    print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

############################################################################################
############################################################################################

"""
############################
Housing Values in Suburbs of Boston

The medv variable is the target variable.
############################
Data description

The Boston data frame has 506 rows and 14 columns.

This data frame contains the following columns:
############################

crim
per capita crime rate by town.

zn
proportion of residential land zoned for lots over 25,000 sq.ft.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built prior to 1940.

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per $10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).

medv
median value of owner-occupied homes in $1000s.
############################
"""

"""
URL: https://towardsdatascience.com/perceptrons-logical-functions-and-the-xor-problem-37ca5025790a
(solving XOR with perceptrons)
"""
