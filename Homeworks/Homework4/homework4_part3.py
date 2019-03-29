"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-1-19

Homework 4 - Classification
Keras Fashion MNIST Dataset - Keras-based Convolutional Neural Network

Notes:

TODO - modify from basic tutorial template to be more robust.

############################################

Resources Used:

URL: https://www.tensorflow.org/tutorials/keras/basic_classification
(Keras tutorial)

URL: https://developers.google.com/machine-learning/practica/image-classification/

############################################

Assignment Instructions:

For this homework, do the following things:

Speculate on whether you believe that so-called “deep” neural networks are destined to be another bust just as
perceptrons and expert systems were in the past, or whether they really are a breakthrough that will be used for
years into the future. Please give a two-to-three-paragraph answer, including examples to back up your argument.

Hand-compute a single, complete backpropagation cycle. Use the example network from class and compute the updated
weight values for the first gradient descent iteration for the AND example, i.e., [1, 1] → 0. Use the same initial
weights we used in the class example.

Build a Keras-based ConvNet for Keras’s Fashion MNIST dataset (fashion_mnist). Experiment with different network
architectures, submit your most performant network, and report the results.

"""
############################################################################################

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

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

train_images = train_images / 255.0

test_images = test_images / 255.0

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

############################################################################################
"""
tf.keras.layers.Flatten - Flattens the input. Does not affect the batch size. (2-d to 1-d array)

tf.keras.layers.Dense - densely-connected, or fully-connected, neural layers.

10-node softmax layer—this returns an array of 10 probability scores that sum to 1
"""
# Setup the layers.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
Loss function —This measures how accurate the model is during training. 
We want to minimize this function to "steer" the model in the right direction.

Optimizer —This is how the model is updated based on the data it sees and its loss function.

Metrics —Used to monitor the training and testing steps. The following example uses accuracy, 
the fraction of the images that are correctly classified
"""
# Compile the model.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Feed the training data to the model—in this example, the train_images and train_labels arrays.

The model learns to associate images and labels.

We ask the model to make predictions about a test set—in this example, the test_images array. 
We verify that the predictions match the labels from the test_labels array.
"""
# Train the model.
model.fit(train_images, train_labels, epochs=5)

############################################################################################
"""
Evaluate accuracy.
"""

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

############################################################################################
"""
Make predictions.
"""
predictions = model.predict(test_images)

# Prediction sample
print("Prediction sample:")
print(predictions[0])

# Get highest confidence value from prediction array.
print("Highest class confidence value:")
print(np.argmax(predictions[0]))

# Confirm against test associated test label.
print("Associated test label value:")
print(test_labels[0])


############################################################################################

def plot_image(i, predictions_array, true_label, img):
    """
    Function outputs a graph of the image.

    :param i:
    :param predictions_array:
    :param true_label:
    :param img:
    :return:
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
                                         class_names[true_label]),
               color=color)


############################################################################################

def plot_value_array(i, predictions_array, true_label):
    """
    Function outputs the prediction array with set of confidence values associated with classes.

    :param i:
    :param predictions_array:
    :param true_label:
    :return:
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


############################################################################################

"""
Main function.  Execute the program.
"""
if __name__ == '__main__':
    print()

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

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

    # Grab an image from the test dataset
    img = test_images[0]

    print("Image shape: ")
    print(img.shape)
    print()

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    print("Image batch shape: ")
    print(img.shape)
    print()

    # Predict the images in the batch.
    predictions_single = model.predict(img)

    print("Image batch prediction results: ")
    print(predictions_single)
    print()

    # Visualize the results of the batch image predictions.
    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    # Get the max confidence value result signifying which class it belongs to in the labels.
    np.argmax(predictions_single[0])

############################################################################################

"""
Console Output:

D:\Dropbox\cs344-ai\venv3.6-64bit\Scripts\python.exe D:/Dropbox/cs344-ai/cs344/Homeworks/Homework4/homework4_part3.py
1.13.1
Training data shape:
(60000, 28, 28)
Training targets shape:
(60000,)
Testing data shape:
(10000, 28, 28)
Testing targets shape:
(10000,)

Training images[0]:
WARNING:tensorflow:From D:\Dropbox\cs344-ai\venv3.6-64bit\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
Epoch 1/5

   32/60000 [..............................] - ETA: 3:35 - loss: 2.6576 - acc: 0.0000e+00
 1088/60000 [..............................] - ETA: 8s - loss: 1.3657 - acc: 0.5423      
 2208/60000 [>.............................] - ETA: 5s - loss: 1.0769 - acc: 0.6336
 3296/60000 [>.............................] - ETA: 4s - loss: 0.9689 - acc: 0.6714
 4416/60000 [=>............................] - ETA: 4s - loss: 0.8980 - acc: 0.6947
 5536/60000 [=>............................] - ETA: 3s - loss: 0.8377 - acc: 0.7141
 6624/60000 [==>...........................] - ETA: 3s - loss: 0.8027 - acc: 0.7249
 7744/60000 [==>...........................] - ETA: 3s - loss: 0.7667 - acc: 0.7357
 8928/60000 [===>..........................] - ETA: 3s - loss: 0.7328 - acc: 0.7469
10080/60000 [====>.........................] - ETA: 2s - loss: 0.7191 - acc: 0.7521
11168/60000 [====>.........................] - ETA: 2s - loss: 0.7018 - acc: 0.7581
12128/60000 [=====>........................] - ETA: 2s - loss: 0.6885 - acc: 0.7634
13184/60000 [=====>........................] - ETA: 2s - loss: 0.6726 - acc: 0.7696
14336/60000 [======>.......................] - ETA: 2s - loss: 0.6607 - acc: 0.7732
15488/60000 [======>.......................] - ETA: 2s - loss: 0.6521 - acc: 0.7761
16608/60000 [=======>......................] - ETA: 2s - loss: 0.6444 - acc: 0.7778
17760/60000 [=======>......................] - ETA: 2s - loss: 0.6371 - acc: 0.7805
18880/60000 [========>.....................] - ETA: 2s - loss: 0.6281 - acc: 0.7835
20000/60000 [=========>....................] - ETA: 2s - loss: 0.6220 - acc: 0.7858
21120/60000 [=========>....................] - ETA: 2s - loss: 0.6147 - acc: 0.7889
22208/60000 [==========>...................] - ETA: 1s - loss: 0.6097 - acc: 0.7903
23264/60000 [==========>...................] - ETA: 1s - loss: 0.6047 - acc: 0.7921
24416/60000 [===========>..................] - ETA: 1s - loss: 0.5978 - acc: 0.7946
25472/60000 [===========>..................] - ETA: 1s - loss: 0.5924 - acc: 0.7968
26624/60000 [============>.................] - ETA: 1s - loss: 0.5856 - acc: 0.7988
27776/60000 [============>.................] - ETA: 1s - loss: 0.5804 - acc: 0.8002
28928/60000 [=============>................] - ETA: 1s - loss: 0.5771 - acc: 0.8013
30080/60000 [==============>...............] - ETA: 1s - loss: 0.5730 - acc: 0.8024
31200/60000 [==============>...............] - ETA: 1s - loss: 0.5682 - acc: 0.8036
32288/60000 [===============>..............] - ETA: 1s - loss: 0.5634 - acc: 0.8057
33408/60000 [===============>..............] - ETA: 1s - loss: 0.5590 - acc: 0.8069
34496/60000 [================>.............] - ETA: 1s - loss: 0.5553 - acc: 0.8083
35584/60000 [================>.............] - ETA: 1s - loss: 0.5519 - acc: 0.8095
36672/60000 [=================>............] - ETA: 1s - loss: 0.5497 - acc: 0.8106
37760/60000 [=================>............] - ETA: 1s - loss: 0.5456 - acc: 0.8118
38784/60000 [==================>...........] - ETA: 1s - loss: 0.5423 - acc: 0.8129
39840/60000 [==================>...........] - ETA: 0s - loss: 0.5390 - acc: 0.8141
40928/60000 [===================>..........] - ETA: 0s - loss: 0.5354 - acc: 0.8151
42048/60000 [====================>.........] - ETA: 0s - loss: 0.5341 - acc: 0.8153
43200/60000 [====================>.........] - ETA: 0s - loss: 0.5319 - acc: 0.8159
44288/60000 [=====================>........] - ETA: 0s - loss: 0.5282 - acc: 0.8171
45280/60000 [=====================>........] - ETA: 0s - loss: 0.5254 - acc: 0.8181
46336/60000 [======================>.......] - ETA: 0s - loss: 0.5228 - acc: 0.8189
47232/60000 [======================>.......] - ETA: 0s - loss: 0.5207 - acc: 0.8195
48160/60000 [=======================>......] - ETA: 0s - loss: 0.5182 - acc: 0.8203
49120/60000 [=======================>......] - ETA: 0s - loss: 0.5169 - acc: 0.8207
50208/60000 [========================>.....] - ETA: 0s - loss: 0.5155 - acc: 0.8210
51264/60000 [========================>.....] - ETA: 0s - loss: 0.5130 - acc: 0.8220
52288/60000 [=========================>....] - ETA: 0s - loss: 0.5117 - acc: 0.8225
53248/60000 [=========================>....] - ETA: 0s - loss: 0.5096 - acc: 0.8234
54272/60000 [==========================>...] - ETA: 0s - loss: 0.5074 - acc: 0.8240
55328/60000 [==========================>...] - ETA: 0s - loss: 0.5053 - acc: 0.8247
56448/60000 [===========================>..] - ETA: 0s - loss: 0.5036 - acc: 0.8252
57568/60000 [===========================>..] - ETA: 0s - loss: 0.5020 - acc: 0.8254
58688/60000 [============================>.] - ETA: 0s - loss: 0.5007 - acc: 0.8256
59840/60000 [============================>.] - ETA: 0s - loss: 0.4988 - acc: 0.8263
60000/60000 [==============================] - 3s 49us/sample - loss: 0.4987 - acc: 0.8263
Epoch 2/5

   32/60000 [..............................] - ETA: 7s - loss: 0.4014 - acc: 0.8438
 1056/60000 [..............................] - ETA: 3s - loss: 0.3558 - acc: 0.8693
 1920/60000 [..............................] - ETA: 3s - loss: 0.3678 - acc: 0.8661
 2912/60000 [>.............................] - ETA: 3s - loss: 0.3734 - acc: 0.8620
 3904/60000 [>.............................] - ETA: 3s - loss: 0.3804 - acc: 0.8594
 4928/60000 [=>............................] - ETA: 2s - loss: 0.3814 - acc: 0.8600
 5984/60000 [=>............................] - ETA: 2s - loss: 0.3861 - acc: 0.8598
 6944/60000 [==>...........................] - ETA: 2s - loss: 0.3889 - acc: 0.8574
 8064/60000 [===>..........................] - ETA: 2s - loss: 0.3874 - acc: 0.8588
 9120/60000 [===>..........................] - ETA: 2s - loss: 0.3880 - acc: 0.8580
10080/60000 [====>.........................] - ETA: 2s - loss: 0.3929 - acc: 0.8563
11104/60000 [====>.........................] - ETA: 2s - loss: 0.3935 - acc: 0.8563
12192/60000 [=====>........................] - ETA: 2s - loss: 0.3927 - acc: 0.8579
13280/60000 [=====>........................] - ETA: 2s - loss: 0.3923 - acc: 0.8584
14400/60000 [======>.......................] - ETA: 2s - loss: 0.3921 - acc: 0.8583
15520/60000 [======>.......................] - ETA: 2s - loss: 0.3903 - acc: 0.8599
16672/60000 [=======>......................] - ETA: 2s - loss: 0.3906 - acc: 0.8598
17760/60000 [=======>......................] - ETA: 2s - loss: 0.3892 - acc: 0.8596
18784/60000 [========>.....................] - ETA: 2s - loss: 0.3871 - acc: 0.8602
19808/60000 [========>.....................] - ETA: 1s - loss: 0.3872 - acc: 0.8602
20896/60000 [=========>....................] - ETA: 1s - loss: 0.3880 - acc: 0.8604
21952/60000 [=========>....................] - ETA: 1s - loss: 0.3900 - acc: 0.8596
23072/60000 [==========>...................] - ETA: 1s - loss: 0.3863 - acc: 0.8608
24224/60000 [===========>..................] - ETA: 1s - loss: 0.3852 - acc: 0.8608
25344/60000 [===========>..................] - ETA: 1s - loss: 0.3858 - acc: 0.8605
26464/60000 [============>.................] - ETA: 1s - loss: 0.3859 - acc: 0.8603
27552/60000 [============>.................] - ETA: 1s - loss: 0.3846 - acc: 0.8605
28448/60000 [=============>................] - ETA: 1s - loss: 0.3837 - acc: 0.8610
29376/60000 [=============>................] - ETA: 1s - loss: 0.3831 - acc: 0.8612
30464/60000 [==============>...............] - ETA: 1s - loss: 0.3830 - acc: 0.8608
31584/60000 [==============>...............] - ETA: 1s - loss: 0.3827 - acc: 0.8609
32704/60000 [===============>..............] - ETA: 1s - loss: 0.3834 - acc: 0.8608
33824/60000 [===============>..............] - ETA: 1s - loss: 0.3835 - acc: 0.8607
34880/60000 [================>.............] - ETA: 1s - loss: 0.3830 - acc: 0.8610
35776/60000 [================>.............] - ETA: 1s - loss: 0.3837 - acc: 0.8610
36736/60000 [=================>............] - ETA: 1s - loss: 0.3833 - acc: 0.8609
37760/60000 [=================>............] - ETA: 1s - loss: 0.3827 - acc: 0.8612
38752/60000 [==================>...........] - ETA: 1s - loss: 0.3819 - acc: 0.8615
39840/60000 [==================>...........] - ETA: 0s - loss: 0.3806 - acc: 0.8621
40896/60000 [===================>..........] - ETA: 0s - loss: 0.3806 - acc: 0.8621
41984/60000 [===================>..........] - ETA: 0s - loss: 0.3795 - acc: 0.8625
43136/60000 [====================>.........] - ETA: 0s - loss: 0.3802 - acc: 0.8628
44256/60000 [=====================>........] - ETA: 0s - loss: 0.3799 - acc: 0.8629
45376/60000 [=====================>........] - ETA: 0s - loss: 0.3794 - acc: 0.8627
46528/60000 [======================>.......] - ETA: 0s - loss: 0.3800 - acc: 0.8626
47584/60000 [======================>.......] - ETA: 0s - loss: 0.3793 - acc: 0.8627
48576/60000 [=======================>......] - ETA: 0s - loss: 0.3792 - acc: 0.8628
49600/60000 [=======================>......] - ETA: 0s - loss: 0.3788 - acc: 0.8630
50592/60000 [========================>.....] - ETA: 0s - loss: 0.3780 - acc: 0.8634
51712/60000 [========================>.....] - ETA: 0s - loss: 0.3774 - acc: 0.8634
52800/60000 [=========================>....] - ETA: 0s - loss: 0.3766 - acc: 0.8637
53888/60000 [=========================>....] - ETA: 0s - loss: 0.3765 - acc: 0.8638
54976/60000 [==========================>...] - ETA: 0s - loss: 0.3762 - acc: 0.8639
56128/60000 [===========================>..] - ETA: 0s - loss: 0.3760 - acc: 0.8641
57216/60000 [===========================>..] - ETA: 0s - loss: 0.3755 - acc: 0.8642
58336/60000 [============================>.] - ETA: 0s - loss: 0.3749 - acc: 0.8642
59392/60000 [============================>.] - ETA: 0s - loss: 0.3750 - acc: 0.8643
60000/60000 [==============================] - 3s 48us/sample - loss: 0.3748 - acc: 0.8644
Epoch 3/5

   32/60000 [..............................] - ETA: 7s - loss: 0.3910 - acc: 0.8125
 1152/60000 [..............................] - ETA: 2s - loss: 0.3248 - acc: 0.8906
 2240/60000 [>.............................] - ETA: 2s - loss: 0.3230 - acc: 0.8879
 3264/60000 [>.............................] - ETA: 2s - loss: 0.3211 - acc: 0.8888
 4384/60000 [=>............................] - ETA: 2s - loss: 0.3264 - acc: 0.8853
 5536/60000 [=>............................] - ETA: 2s - loss: 0.3283 - acc: 0.8844
 6656/60000 [==>...........................] - ETA: 2s - loss: 0.3269 - acc: 0.8845
 7776/60000 [==>...........................] - ETA: 2s - loss: 0.3277 - acc: 0.8846
 8928/60000 [===>..........................] - ETA: 2s - loss: 0.3328 - acc: 0.8814
10048/60000 [====>.........................] - ETA: 2s - loss: 0.3302 - acc: 0.8828
11104/60000 [====>.........................] - ETA: 2s - loss: 0.3314 - acc: 0.8828
12192/60000 [=====>........................] - ETA: 2s - loss: 0.3362 - acc: 0.8804
13152/60000 [=====>........................] - ETA: 2s - loss: 0.3366 - acc: 0.8797
14304/60000 [======>.......................] - ETA: 2s - loss: 0.3341 - acc: 0.8805
15456/60000 [======>.......................] - ETA: 2s - loss: 0.3346 - acc: 0.8808
16512/60000 [=======>......................] - ETA: 2s - loss: 0.3379 - acc: 0.8795
17664/60000 [=======>......................] - ETA: 1s - loss: 0.3392 - acc: 0.8792
18816/60000 [========>.....................] - ETA: 1s - loss: 0.3430 - acc: 0.8781
19968/60000 [========>.....................] - ETA: 1s - loss: 0.3427 - acc: 0.8781
21120/60000 [=========>....................] - ETA: 1s - loss: 0.3420 - acc: 0.8788
22240/60000 [==========>...................] - ETA: 1s - loss: 0.3427 - acc: 0.8782
23392/60000 [==========>...................] - ETA: 1s - loss: 0.3425 - acc: 0.8782
24544/60000 [===========>..................] - ETA: 1s - loss: 0.3437 - acc: 0.8774
25600/60000 [===========>..................] - ETA: 1s - loss: 0.3438 - acc: 0.8772
26752/60000 [============>.................] - ETA: 1s - loss: 0.3432 - acc: 0.8775
27904/60000 [============>.................] - ETA: 1s - loss: 0.3451 - acc: 0.8768
29088/60000 [=============>................] - ETA: 1s - loss: 0.3446 - acc: 0.8769
30208/60000 [==============>...............] - ETA: 1s - loss: 0.3447 - acc: 0.8768
31328/60000 [==============>...............] - ETA: 1s - loss: 0.3457 - acc: 0.8763
32480/60000 [===============>..............] - ETA: 1s - loss: 0.3445 - acc: 0.8767
33568/60000 [===============>..............] - ETA: 1s - loss: 0.3449 - acc: 0.8764
34656/60000 [================>.............] - ETA: 1s - loss: 0.3432 - acc: 0.8769
35776/60000 [================>.............] - ETA: 1s - loss: 0.3419 - acc: 0.8775
36800/60000 [=================>............] - ETA: 1s - loss: 0.3407 - acc: 0.8779
37920/60000 [=================>............] - ETA: 1s - loss: 0.3393 - acc: 0.8780
39072/60000 [==================>...........] - ETA: 0s - loss: 0.3389 - acc: 0.8780
40224/60000 [===================>..........] - ETA: 0s - loss: 0.3394 - acc: 0.8777
41376/60000 [===================>..........] - ETA: 0s - loss: 0.3387 - acc: 0.8779
42560/60000 [====================>.........] - ETA: 0s - loss: 0.3394 - acc: 0.8777
43680/60000 [====================>.........] - ETA: 0s - loss: 0.3396 - acc: 0.8774
44800/60000 [=====================>........] - ETA: 0s - loss: 0.3400 - acc: 0.8770
45952/60000 [=====================>........] - ETA: 0s - loss: 0.3400 - acc: 0.8773
46976/60000 [======================>.......] - ETA: 0s - loss: 0.3407 - acc: 0.8768
47904/60000 [======================>.......] - ETA: 0s - loss: 0.3403 - acc: 0.8770
48992/60000 [=======================>......] - ETA: 0s - loss: 0.3395 - acc: 0.8770
50080/60000 [========================>.....] - ETA: 0s - loss: 0.3399 - acc: 0.8768
51200/60000 [========================>.....] - ETA: 0s - loss: 0.3402 - acc: 0.8768
52320/60000 [=========================>....] - ETA: 0s - loss: 0.3401 - acc: 0.8767
53440/60000 [=========================>....] - ETA: 0s - loss: 0.3398 - acc: 0.8765
54560/60000 [==========================>...] - ETA: 0s - loss: 0.3394 - acc: 0.8766
55616/60000 [==========================>...] - ETA: 0s - loss: 0.3393 - acc: 0.8767
56768/60000 [===========================>..] - ETA: 0s - loss: 0.3389 - acc: 0.8766
57920/60000 [===========================>..] - ETA: 0s - loss: 0.3373 - acc: 0.8773
58976/60000 [============================>.] - ETA: 0s - loss: 0.3371 - acc: 0.8772
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3372 - acc: 0.8772
Epoch 4/5

   32/60000 [..............................] - ETA: 7s - loss: 0.4595 - acc: 0.7812
  992/60000 [..............................] - ETA: 3s - loss: 0.3199 - acc: 0.8800
 2016/60000 [>.............................] - ETA: 3s - loss: 0.3135 - acc: 0.8834
 3136/60000 [>.............................] - ETA: 2s - loss: 0.3167 - acc: 0.8846
 4288/60000 [=>............................] - ETA: 2s - loss: 0.3172 - acc: 0.8827
 5376/60000 [=>............................] - ETA: 2s - loss: 0.3164 - acc: 0.8817
 6528/60000 [==>...........................] - ETA: 2s - loss: 0.3168 - acc: 0.8804
 7680/60000 [==>...........................] - ETA: 2s - loss: 0.3148 - acc: 0.8819
 8800/60000 [===>..........................] - ETA: 2s - loss: 0.3165 - acc: 0.8816
 9888/60000 [===>..........................] - ETA: 2s - loss: 0.3150 - acc: 0.8824
11040/60000 [====>.........................] - ETA: 2s - loss: 0.3150 - acc: 0.8832
12192/60000 [=====>........................] - ETA: 2s - loss: 0.3157 - acc: 0.8838
13376/60000 [=====>........................] - ETA: 2s - loss: 0.3177 - acc: 0.8827
14496/60000 [======>.......................] - ETA: 2s - loss: 0.3158 - acc: 0.8829
15616/60000 [======>.......................] - ETA: 2s - loss: 0.3166 - acc: 0.8831
16768/60000 [=======>......................] - ETA: 1s - loss: 0.3182 - acc: 0.8822
17888/60000 [=======>......................] - ETA: 1s - loss: 0.3184 - acc: 0.8823
19072/60000 [========>.....................] - ETA: 1s - loss: 0.3178 - acc: 0.8828
20192/60000 [=========>....................] - ETA: 1s - loss: 0.3181 - acc: 0.8828
21312/60000 [=========>....................] - ETA: 1s - loss: 0.3189 - acc: 0.8828
22336/60000 [==========>...................] - ETA: 1s - loss: 0.3200 - acc: 0.8823
23488/60000 [==========>...................] - ETA: 1s - loss: 0.3213 - acc: 0.8816
24512/60000 [===========>..................] - ETA: 1s - loss: 0.3224 - acc: 0.8812
25568/60000 [===========>..................] - ETA: 1s - loss: 0.3224 - acc: 0.8812
26720/60000 [============>.................] - ETA: 1s - loss: 0.3219 - acc: 0.8820
27872/60000 [============>.................] - ETA: 1s - loss: 0.3210 - acc: 0.8822
28992/60000 [=============>................] - ETA: 1s - loss: 0.3215 - acc: 0.8820
30048/60000 [==============>...............] - ETA: 1s - loss: 0.3206 - acc: 0.8825
31168/60000 [==============>...............] - ETA: 1s - loss: 0.3187 - acc: 0.8833
32352/60000 [===============>..............] - ETA: 1s - loss: 0.3194 - acc: 0.8828
33472/60000 [===============>..............] - ETA: 1s - loss: 0.3188 - acc: 0.8831
34592/60000 [================>.............] - ETA: 1s - loss: 0.3181 - acc: 0.8832
35776/60000 [================>.............] - ETA: 1s - loss: 0.3179 - acc: 0.8833
36928/60000 [=================>............] - ETA: 1s - loss: 0.3181 - acc: 0.8832
38048/60000 [==================>...........] - ETA: 1s - loss: 0.3175 - acc: 0.8836
39136/60000 [==================>...........] - ETA: 0s - loss: 0.3179 - acc: 0.8831
40288/60000 [===================>..........] - ETA: 0s - loss: 0.3187 - acc: 0.8832
41472/60000 [===================>..........] - ETA: 0s - loss: 0.3190 - acc: 0.8831
42624/60000 [====================>.........] - ETA: 0s - loss: 0.3190 - acc: 0.8833
43744/60000 [====================>.........] - ETA: 0s - loss: 0.3178 - acc: 0.8837
44864/60000 [=====================>........] - ETA: 0s - loss: 0.3168 - acc: 0.8838
45888/60000 [=====================>........] - ETA: 0s - loss: 0.3162 - acc: 0.8838
47008/60000 [======================>.......] - ETA: 0s - loss: 0.3163 - acc: 0.8838
48128/60000 [=======================>......] - ETA: 0s - loss: 0.3162 - acc: 0.8839
49280/60000 [=======================>......] - ETA: 0s - loss: 0.3150 - acc: 0.8843
50432/60000 [========================>.....] - ETA: 0s - loss: 0.3153 - acc: 0.8843
51520/60000 [========================>.....] - ETA: 0s - loss: 0.3149 - acc: 0.8844
52608/60000 [=========================>....] - ETA: 0s - loss: 0.3148 - acc: 0.8844
53728/60000 [=========================>....] - ETA: 0s - loss: 0.3145 - acc: 0.8846
54912/60000 [==========================>...] - ETA: 0s - loss: 0.3146 - acc: 0.8847
56064/60000 [===========================>..] - ETA: 0s - loss: 0.3146 - acc: 0.8846
57216/60000 [===========================>..] - ETA: 0s - loss: 0.3140 - acc: 0.8849
58368/60000 [============================>.] - ETA: 0s - loss: 0.3147 - acc: 0.8846
59520/60000 [============================>.] - ETA: 0s - loss: 0.3142 - acc: 0.8846
60000/60000 [==============================] - 3s 46us/sample - loss: 0.3141 - acc: 0.8846
Epoch 5/5

   32/60000 [..............................] - ETA: 5s - loss: 0.1405 - acc: 0.9688
 1120/60000 [..............................] - ETA: 2s - loss: 0.2985 - acc: 0.8991
 2208/60000 [>.............................] - ETA: 2s - loss: 0.2926 - acc: 0.8958
 3360/60000 [>.............................] - ETA: 2s - loss: 0.2984 - acc: 0.8946
 4544/60000 [=>............................] - ETA: 2s - loss: 0.2918 - acc: 0.8957
 5696/60000 [=>............................] - ETA: 2s - loss: 0.2920 - acc: 0.8926
 6816/60000 [==>...........................] - ETA: 2s - loss: 0.2877 - acc: 0.8941
 7904/60000 [==>...........................] - ETA: 2s - loss: 0.2918 - acc: 0.8921
 8928/60000 [===>..........................] - ETA: 2s - loss: 0.2910 - acc: 0.8920
10112/60000 [====>.........................] - ETA: 2s - loss: 0.2936 - acc: 0.8903
11200/60000 [====>.........................] - ETA: 2s - loss: 0.2938 - acc: 0.8904
12192/60000 [=====>........................] - ETA: 2s - loss: 0.2955 - acc: 0.8904
13216/60000 [=====>........................] - ETA: 2s - loss: 0.2928 - acc: 0.8914
14240/60000 [======>.......................] - ETA: 2s - loss: 0.2932 - acc: 0.8908
15200/60000 [======>.......................] - ETA: 2s - loss: 0.2942 - acc: 0.8902
16224/60000 [=======>......................] - ETA: 2s - loss: 0.2951 - acc: 0.8899
17312/60000 [=======>......................] - ETA: 2s - loss: 0.2970 - acc: 0.8895
18464/60000 [========>.....................] - ETA: 1s - loss: 0.2955 - acc: 0.8897
19584/60000 [========>.....................] - ETA: 1s - loss: 0.2964 - acc: 0.8903
20704/60000 [=========>....................] - ETA: 1s - loss: 0.2954 - acc: 0.8907
21856/60000 [=========>....................] - ETA: 1s - loss: 0.2955 - acc: 0.8904
23008/60000 [==========>...................] - ETA: 1s - loss: 0.2949 - acc: 0.8910
24096/60000 [===========>..................] - ETA: 1s - loss: 0.2957 - acc: 0.8906
25248/60000 [===========>..................] - ETA: 1s - loss: 0.2963 - acc: 0.8908
26400/60000 [============>.................] - ETA: 1s - loss: 0.2960 - acc: 0.8909
27488/60000 [============>.................] - ETA: 1s - loss: 0.2952 - acc: 0.8909
28512/60000 [=============>................] - ETA: 1s - loss: 0.2942 - acc: 0.8913
29632/60000 [=============>................] - ETA: 1s - loss: 0.2942 - acc: 0.8916
30656/60000 [==============>...............] - ETA: 1s - loss: 0.2945 - acc: 0.8918
31680/60000 [==============>...............] - ETA: 1s - loss: 0.2944 - acc: 0.8913
32736/60000 [===============>..............] - ETA: 1s - loss: 0.2944 - acc: 0.8914
33728/60000 [===============>..............] - ETA: 1s - loss: 0.2937 - acc: 0.8916
34720/60000 [================>.............] - ETA: 1s - loss: 0.2936 - acc: 0.8916
35744/60000 [================>.............] - ETA: 1s - loss: 0.2932 - acc: 0.8919
36832/60000 [=================>............] - ETA: 1s - loss: 0.2931 - acc: 0.8919
37920/60000 [=================>............] - ETA: 1s - loss: 0.2926 - acc: 0.8920
39040/60000 [==================>...........] - ETA: 0s - loss: 0.2935 - acc: 0.8917
40192/60000 [===================>..........] - ETA: 0s - loss: 0.2939 - acc: 0.8918
41312/60000 [===================>..........] - ETA: 0s - loss: 0.2948 - acc: 0.8913
42432/60000 [====================>.........] - ETA: 0s - loss: 0.2958 - acc: 0.8912
43520/60000 [====================>.........] - ETA: 0s - loss: 0.2964 - acc: 0.8911
44672/60000 [=====================>........] - ETA: 0s - loss: 0.2975 - acc: 0.8907
45824/60000 [=====================>........] - ETA: 0s - loss: 0.2964 - acc: 0.8912
46976/60000 [======================>.......] - ETA: 0s - loss: 0.2965 - acc: 0.8912
48128/60000 [=======================>......] - ETA: 0s - loss: 0.2963 - acc: 0.8913
49280/60000 [=======================>......] - ETA: 0s - loss: 0.2966 - acc: 0.8916
50400/60000 [========================>.....] - ETA: 0s - loss: 0.2959 - acc: 0.8919
51552/60000 [========================>.....] - ETA: 0s - loss: 0.2956 - acc: 0.8920
52736/60000 [=========================>....] - ETA: 0s - loss: 0.2959 - acc: 0.8919
53760/60000 [=========================>....] - ETA: 0s - loss: 0.2956 - acc: 0.8921
54848/60000 [==========================>...] - ETA: 0s - loss: 0.2952 - acc: 0.8920
56000/60000 [===========================>..] - ETA: 0s - loss: 0.2950 - acc: 0.8920
57120/60000 [===========================>..] - ETA: 0s - loss: 0.2956 - acc: 0.8916
58272/60000 [============================>.] - ETA: 0s - loss: 0.2951 - acc: 0.8920
59360/60000 [============================>.] - ETA: 0s - loss: 0.2948 - acc: 0.8922
60000/60000 [==============================] - 3s 47us/sample - loss: 0.2954 - acc: 0.8919

   32/10000 [..............................] - ETA: 13s - loss: 0.4000 - acc: 0.8438
 2848/10000 [=======>......................] - ETA: 0s - loss: 0.3284 - acc: 0.8754 
 5728/10000 [================>.............] - ETA: 0s - loss: 0.3634 - acc: 0.8673
 8576/10000 [========================>.....] - ETA: 0s - loss: 0.3506 - acc: 0.8702
10000/10000 [==============================] - 0s 22us/sample - loss: 0.3480 - acc: 0.8718
Test accuracy: 0.8718
Prediction sample:
[6.2122416e-07 1.5767617e-08 1.1753525e-06 3.3560596e-07 5.9467931e-07
 1.9601842e-02 1.3582300e-06 7.2436824e-02 1.7854398e-03 9.0617180e-01]
Highest class confidence value:
9
Associated test label value:
9

Image shape: 
(28, 28)

Image batch shape: 
(1, 28, 28)

Image batch prediction results: 
[[6.2122592e-07 1.5767617e-08 1.1753547e-06 3.3560664e-07 5.9468044e-07
  1.9601863e-02 1.3582312e-06 7.2436847e-02 1.7854432e-03 9.0617180e-01]]


Process finished with exit code 0

"""
