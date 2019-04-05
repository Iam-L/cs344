"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-1-19

Homework 4 - Classification
Keras Fashion MNIST Dataset - Keras-based Convolutional Neural Network

Notes:

I took material from two tutorials and sort of meshed them together.

I don't pretend to understand everything that's happening, but it seems to be a working CNN.

############################################

Resources Used:

URL: https://www.tensorflow.org/tutorials/keras/basic_classification
(Keras Tensorflow tutorial)

URL: https://developers.google.com/machine-learning/practica/image-classification/
(Google Crash Course)

URL: https://www.markdownguide.org/basic-syntax/
(Markdown syntac for Juypter Notebook)

URL: https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/
(Keras Fashion MNIST Tutorial)

URL: https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/
(Keras Convolutional Layers)

############################################

Assignment Instructions:

Build a Keras-based ConvNet for Keras’s Fashion MNIST dataset (fashion_mnist). Experiment with different network
architectures, submit your most performant network, and report the results.

"""
############################################################################################

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Initialize the number of epochs to train for, base learning rate, and batch size.
NUM_EPOCHS = 25
LEARNING_RATE = 1e-2
BATCH_SIZE = 32

print(tf.__version__)

############################################################################################
"""
Label	Class
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
# Load the dataset.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Column headers.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training data shape:")
print(train_images.shape)
print("Training targets shape:")
print(train_labels.shape)
print("Testing data shape:")
print(test_images.shape)
print("Testing targets shape:")
print(test_labels.shape)
print()

# Display 1st image in training set.
print("Training images[0]:")
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

############################################################################################
"""
Pre-process dataset.
"""

# Scale data to the range of [0, 1].
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Display first 25 images in training set.
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# If we are using "channels first" ordering, then reshape the design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    train_images_reshaped = train_images.reshape((train_images.shape[0], 1, 28, 28))
    test_images_reshaped = test_images.reshape((test_images.shape[0], 1, 28, 28))

# Otherwise, we are using "channels last" ordering, so the design matrix shape should be:
# num_samples x rows x columns x depth
else:
    train_images_reshaped = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images_reshaped = test_images.reshape((test_images.shape[0], 28, 28, 1))

print("Training data shape after reshape:")
print(train_images_reshaped.shape)
print("Testing data shape after reshape:")
print(test_images_reshaped.shape)
print()

# One-hot encode the training and testing labels.
train_labels_one_hot = np_utils.to_categorical(train_labels, 10)
test_labels_one_hot = np_utils.to_categorical(test_labels, 10)

print("Training targets shape after one-hot encoding:")
print(train_labels_one_hot.shape)
print("Testing targets shape after one-hot encoding:")
print(test_labels_one_hot.shape)
print()

############################################################################################
"""
tf.keras.layers.Flatten - Flattens the input. Does not affect the batch size. (2-d to 1-d array)

tf.keras.layers.Dense - densely-connected, or fully-connected, neural layers.

10-node softmax layer—this returns an array of 10 probability scores that sum to 1.

Pooling layers help to progressively reduce the spatial dimensions of the input volume.

Batch normalization seeks to normalize the activations of a given input volume before passing it into the next layer. 
It has been shown to be effective at reducing the number of epochs required to train a CNN at the expense of an 
increase in per-epoch time.

Dropout is a form of regularization that aims to prevent overfitting. 
Random connections are dropped to ensure that no single node in the network is responsible for activating 
when presented with a given pattern.

What follows is a fully-connected layer and softmax classifier (Lines 49-57). 
The softmax classifier is used to obtain output classification probabilities.
"""


def build(width, height, depth, classes):
    """
    Function builds the machine learning. my_model.
    
    :param width:  width of the image file in pixels.
    :param height:  height of the image file in pixels.
    :param depth:  the channels - r,g,b,a.
    :param classes: all the possible labels for each image.
    :return: the constructed machine learning my_model.
    """

    # Initialize the my_model along with the input shape to be "channels last" and the channels dimension itself.
    my_model = Sequential()
    input_shape = (height, width, depth)
    channel_dimensions = -1

    # If we are using "channels first", update the input shape and channels dimension.
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)
        channel_dimensions = 1

    # First CONV => RELU => CONV => RELU => POOL layer set.
    my_model.add(Conv2D(32, (3, 3), padding="same",
                        input_shape=input_shape))
    my_model.add(Activation("relu"))
    my_model.add(BatchNormalization(axis=channel_dimensions))
    my_model.add(Conv2D(32, (3, 3), padding="same"))
    my_model.add(Activation("relu"))
    my_model.add(BatchNormalization(axis=channel_dimensions))
    my_model.add(MaxPooling2D(pool_size=(2, 2)))
    my_model.add(Dropout(0.25))

    # Second CONV => RELU => CONV => RELU => POOL layer set.
    my_model.add(Conv2D(64, (3, 3), padding="same"))
    my_model.add(Activation("relu"))
    my_model.add(BatchNormalization(axis=channel_dimensions))
    my_model.add(Conv2D(64, (3, 3), padding="same"))
    my_model.add(Activation("relu"))
    my_model.add(BatchNormalization(axis=channel_dimensions))
    my_model.add(MaxPooling2D(pool_size=(2, 2)))
    my_model.add(Dropout(0.25))

    # First (and only) set of FC => RELU layers
    my_model.add(Flatten())
    my_model.add(Dense(512))
    my_model.add(Activation("relu"))
    my_model.add(BatchNormalization())
    my_model.add(Dropout(0.5))

    # Softmax classifier.
    my_model.add(Dense(classes))
    my_model.add(Activation("softmax"))

    # Return the constructed network architecture.
    return my_model


"""
Loss function —This measures how accurate the model is during training. 
We want to minimize this function to "steer" the model in the right direction.

Optimizer —This is how the model is updated based on the data it sees and its loss function.

Metrics —Used to monitor the training and testing steps. The following example uses accuracy, 
the fraction of the images that are correctly classified
"""

# Initialize the optimizer and the model.
opt = SGD(lr=LEARNING_RATE, momentum=0.9, decay=LEARNING_RATE / NUM_EPOCHS)
model = build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

"""
Feed the training data to the model—in this example, the train_images and train_labels arrays.

The model learns to associate images and labels.

We ask the model to make predictions about a test set—in this example, the test_images array. 
We verify that the predictions match the labels from the test_labels array.
"""

# Train the model.
train_model = model.fit(train_images_reshaped, train_labels_one_hot,
                        validation_data=(test_images_reshaped, test_labels_one_hot),
                        batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

############################################################################################
"""
Evaluate accuracy.
"""

test_loss, test_acc = model.evaluate(test_images_reshaped, test_labels_one_hot)

print()
print('Test dataset accuracy:', test_acc)
print('Test dataset loss:', test_loss)
print()

############################################################################################
"""
Make predictions.
"""
predictions = model.predict(test_images_reshaped)


############################################################################################

def plot_image(i, predictions_array, true_label, img):
    """
    Function outputs a graph of the image.

    :param i: counter variable.
    :param predictions_array: array of all predictions made.
    :param true_label: actual identity of apparel in image.
    :param img: fashion apparel image.
    :return: nothing.
    """

    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


############################################################################################

def plot_value_array(i, predictions_array, true_label):
    """
    Function outputs the prediction array with set of confidence values associated with each class.

    :param i: counter variable.
    :param predictions_array: array of all predictions made.
    :param true_label: actualy identity of apparel in image.
    :return: nothing.
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    this_plot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


############################################################################################

def visualize_training_results(predictions_array):
    """
    Function visualizes the results of training the model as a summary of statistics and a plot
    of training loss and accuracy on the dataset.

    :return: Nothing.
    """
    # Show a nicely formatted classification report.
    print("[INFO] evaluating network...")
    print(classification_report(test_labels_one_hot.argmax(axis=1), predictions_array.argmax(axis=1),
                                target_names=class_names))

    # Plot the training loss and accuracy (for each epoch).
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), train_model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), train_model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, NUM_EPOCHS), train_model.history["acc"], label="train_acc")
    plt.plot(np.arange(0, NUM_EPOCHS), train_model.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("training_loss_accuracy_plot.png")

    # Plot the first X test images, their predicted label, and the true label.
    # Color correct predictions in blue, incorrect predictions in red.
    num_rows = 10
    num_cols = 5
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


############################################################################################

def single_image_prediction_results():
    """
    Predict probabilities for a single image.

    :return: Nothing.
    """

    import random
    random_image = random.randint(1, 10000)

    # Prediction sample.
    print()
    print("Prediction for the randomly chosen image:")
    print(predictions[random_image])
    print()

    # Get the highest confidence value from the prediction array.
    print("Highest class confidence value for the image:")
    print(np.argmax(predictions[random_image]))

    # Confirm against associated test label.
    print("Associated test label value for the image:")
    print(test_labels[random_image])
    print()

    # Grab an image from the test dataset.
    image = test_images_reshaped[random_image]
    print("Randomly chosen image's shape: ")
    print(image.shape)
    print()
    # Add the image to a batch where it's the only member.
    image = (np.expand_dims(image, 0))
    print("Randomly chosen image's shape in batch it was added to: ")
    print(image.shape)
    print()
    # Predict the images in the batch.
    predictions_single = model.predict(image)
    print("Randomly chosen image's batch prediction results: ")
    print(predictions_single)
    print()

    # Visualize the results of the batch image predictions.
    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    # Get the max confidence value result signifying which class it belongs to in the labels.
    print("Highest class confidence value for the image in the batch:")
    print(np.argmax(predictions_single[0]))

    # Confirm against associated test label.
    print("Associated test label value for the image in the batch:")
    print(test_labels[random_image])
    print()


############################################################################################
"""
Main function.  Execute the program.
"""
# Debug variable.
debug = 0

if __name__ == '__main__':
    print()

    if debug:
        # Visualize 0th image, predictions, and prediction array.
        i = 0
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(i, predictions, test_labels)
        plt.show()

        # Visualize the 12th image, predictions, and prediction array.
        i = 12
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(1, 2, 2)
        plot_value_array(i, predictions, test_labels)
        plt.show()

    ##############################################

    # Predict for a single image.
    single_image_prediction_results()

    ##############################################

    # Visualize the training and prediction results.
    visualize_training_results(predictions)

############################################################################################
