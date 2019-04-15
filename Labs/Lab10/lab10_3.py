"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-31-19
Assignment: Lab 10 - Neural Networks

Notes:

Exercise 10.3 - Classifying Handwritten Digits with Neural Networks

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.
"""

from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/mnist_train_small.csv",
    sep=",",
    header=None)

# Use just the first 10,000 records for training/validation.
mnist_dataframe = mnist_dataframe.head(10000)

mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
print(mnist_dataframe.head())
print()

###########################################################################################

# Print column 72.
print(mnist_dataframe.loc[:, 72:72])
print()


###########################################################################################

def parse_labels_and_features(dataset):
    """Extracts labels and features.

    This is a good place to scale or transform the features if needed.

    Args:
      dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
      A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[0]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:, 1:784]
    # Scale the data to [0, 1] by dividing out the max value, 255.
    features = features / 255

    return labels, features


###########################################################################################

# Define training dataset.
training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
print(training_examples.describe())

# Define validation dataset.
validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])
print(validation_examples.describe())

# Visualize random example.
rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28))
ax.set_title("Label: %i" % training_targets.loc[rand_example])
ax.grid(False)
plt.show()


###########################################################################################

def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
      A set of feature columns
    """

    # There are 784 pixels in each image.
    return set([tf.feature_column.numeric_column('pixels', shape=784)])


###########################################################################################

def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
    """A custom input_fn for sending MNIST data to the estimator for training.

    Args:
      features: The training features.
      labels: The training labels.
      batch_size: Batch size to use during training.

    Returns:
      A function that returns batches of training features and labels during
      training.
    """

    def _input_fn(num_epochs=None, shuffle=True):
        # Input pipelines are reset with each call to .train(). To ensure model
        # gets a good sampling of data, even when number of steps is small, we
        # shuffle all the data before creating the Dataset object
        idx = np.random.permutation(features.index)
        raw_features = {"pixels": features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


###########################################################################################

def create_predict_input_fn(features, labels, batch_size):
    """A custom input_fn for sending mnist data to the estimator for predictions.

    Args:
      features: The features to base predictions on.
      labels: The labels of the prediction examples.

    Returns:
      A function that returns features and labels for predictions.
    """

    def _input_fn():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


###########################################################################################

def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, and a confusion
    matrix.

    Args:
      learning_rate: A `float`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `LinearClassifier` object.
    """

    periods = 10

    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create a LinearClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


###########################################################################################

def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      learning_rate: A `float`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create feature columns.
    feature_columns = [tf.feature_column.numeric_column('pixels', shape=784)]

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


###########################################################################################

"""
Task 1: Build a Linear Model for MNIST

Yep, this takes forever to train on my system, depending on hyper-parameters (small learning rate ftl)
"""

# # Train on a linear classification model.
# classifier = train_linear_classification_model(
#     learning_rate=0.2,
#     steps=1000,
#     batch_size=10,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

"""
Task 2: Replace the Linear Classifier with a Neural Network

Yep, I cheated by looking at the solution for this one.
"""

# Train on a classification neural network model..
classifier_nn = train_nn_classification_model(
    learning_rate=0.2,
    steps=1000,
    batch_size=10,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# Obtain test dataset.
mnist_test_dataframe = pd.read_csv(
    "https://download.mlcc.google.com/mledu-datasets/mnist_test.csv",
    sep=",",
    header=None)

test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)
print(test_examples.describe())
print()

# Obtain predictions.
predict_test_input_fn = create_predict_input_fn(
    test_examples, test_targets, batch_size=100)

test_predictions = classifier_nn.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

accuracy = metrics.accuracy_score(test_targets, test_predictions)
print("Accuracy on test data: %0.2f" % accuracy)

"""
Task 3: Visualize the weights of the first hidden layer.

"""

print(classifier_nn.get_variable_names())

weights0 = classifier_nn.get_variable_value("dnn/hiddenlayer_0/kernel")

print("weights0 shape:", weights0.shape)

num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    # Weights in coef is reshaped from 1x784 to 28x28.
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

###########################################################################################

"""
Exercise 10.3 Questions:
###########################################################################################

Task 1: What does the confusion matrix show for this example?

The confusion matrix shows the number of images that were incorrectly identified as a specific digit 0-9.

##################################################

Task 2: How does the TensorFlow network architecture differ from the Keras example given in class? 
Report any improvements you can make over the baseline testset accuracy for this task.

The TensorFlow network architecture is defined by the DNNClassifier object.
There is a Softmax layer at the top that selects the winning class.
We only modify the regularization method, # of hidden layers and # of nodes in each, along with a few other parameters.
All the lower level details are pretty much hidden.

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)

The Keras network architecture is modular so we can add individual layers one-by-one.  Thus, it is far easier to see
every layer that is in the model.  There is an embedding layer as evidenced by: 

model.add(Embedding(max_features, 32)) <-- recurrent neural network in-class example.
model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) <-- word embeddings in-class example.

There doesn't appear to be an embedding layer in the Tensorflow network from what I can see.

print(classifier.get_variable_names()) <-- prints the hidden layers.

['dnn/hiddenlayer_0/bias', 'dnn/hiddenlayer_0/bias/t_0/Adagrad', 'dnn/hiddenlayer_0/kernel', 
'dnn/hiddenlayer_0/kernel/t_0/Adagrad', 'dnn/hiddenlayer_1/bias', 'dnn/hiddenlayer_1/bias/t_0/Adagrad', 
'dnn/hiddenlayer_1/kernel', 'dnn/hiddenlayer_1/kernel/t_0/Adagrad', 'dnn/logits/bias', 'dnn/logits/bias/t_0/Adagrad', 
'dnn/logits/kernel', 'dnn/logits/kernel/t_0/Adagrad', 'global_step'


Note: I have not made any improvements over the base-line test-set accuracy.

##################################################

Task 3: What differences can you see between the visualizations for 10 steps and 1000 steps?

As the # of steps goes down, the images become a lot noisier and abstract with fewer definable shapes, patterns, etc.

##################################################

Task 1:

# Train on a linear classification model.
classifier = train_linear_classification_model(
    learning_rate=0.2,
    steps=1000,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
Training model...
LogLoss error (on validation data):
  period 00 : 6.94
  period 01 : 4.81
  period 02 : 4.35
  period 03 : 3.97
  period 04 : 4.26
  period 05 : 3.79
  period 06 : 3.90
  period 07 : 3.45
  period 08 : 3.67
  period 09 : 3.45
Model training finished.
Final accuracy (on validation data): 0.90

Process finished with exit code 0

###########################
Task 2 (using provided solution):

classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
Training model...
LogLoss error (on validation data):
  period 00 : 6.31
  period 01 : 3.80
  period 02 : 2.98
  period 03 : 3.12
  period 04 : 2.94
  period 05 : 2.64
  period 06 : 2.46
  period 07 : 2.51
  period 08 : 1.89
  period 09 : 1.88
Model training finished.
Final accuracy (on validation data): 0.95

Accuracy on test data: 0.95

###########################
Task 3:

classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=100,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
Training model...
LogLoss error (on validation data):
  period 00 : 14.82
  period 01 : 10.79
  period 02 : 7.42
  period 03 : 5.97
  period 04 : 8.95
  period 05 : 5.73
  period 06 : 7.11
  period 07 : 5.75
  period 08 : 4.57
  period 09 : 4.52
Model training finished.
Final accuracy (on validation data): 0.87

Accuracy on test data: 0.88

classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=10,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
    
Training model...
LogLoss error (on validation data):
  period 00 : 22.88
  period 01 : 21.87
  period 02 : 22.53
  period 03 : 19.00
  period 04 : 21.04
  period 05 : 16.87
  period 06 : 16.56
  period 07 : 19.74
  period 08 : 12.09
  period 09 : 18.65
Model training finished.
Final accuracy (on validation data): 0.46

Accuracy on test data: 0.46

"""
