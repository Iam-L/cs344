"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-27-19
Assignment: Lab 09 - Classification

Notes:

Exercise 9.3 - ML Practicum: Image Classification

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.

PyCharm gives import errors but everything still seems to work fine.

FIXME - error in the visualize intermediate representation matplotlib code in trying to run locally.
"""

###########################################################################################
import os
import time
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import RMSprop

###########################################################################################

# Extract contents of zip file.
# local_zip = 'cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp')
# zip_ref.close()

###########################################################################################

"""
Configure random transformations for data augmentation.
"""

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

###########################################################################################

# Define directories.
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Print filenames in directories.
train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])

train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames.sort()
print(train_dog_fnames[:10])

# Print # of available images in directories.
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))

###########################################################################################

"""
Configure matplotlib parameters and display images.
Note: Requires Pillow package to be installed.
"""

# %matplotlib inline

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

###########################################################################################
"""
View datagen transformations on cat images.
"""

img_path = os.path.join(train_cats_dir, train_cat_fnames[2])
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# The .flow() command below generates batches of randomly transformed images
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(array_to_img(batch[0]))
    i += 1
    if i % 5 == 0:
        break

plt.show()

###########################################################################################

"""
CNN architecture. (modified to use dropout)
"""

# # Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# # the three color channels: R, G, and B
# img_input = layers.Input(shape=(150, 150, 3))
#
# # First convolution extracts 16 filters that are 3x3
# # Convolution is followed by max-pooling layer with a 2x2 window
# x = layers.Conv2D(16, 3, activation='relu')(img_input)
# x = layers.MaxPooling2D(2)(x)
#
# # Second convolution extracts 32 filters that are 3x3
# # Convolution is followed by max-pooling layer with a 2x2 window
# x = layers.Conv2D(32, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)
#
# # Third convolution extracts 64 filters that are 3x3
# # Convolution is followed by max-pooling layer with a 2x2 window
# x = layers.Conv2D(64, 3, activation='relu')(x)
# x = layers.MaxPooling2D(2)(x)
#
# # Flatten feature map to a 1-dim tensor so we can add fully connected layers
# x = layers.Flatten()(x)
#
# # Create a fully connected layer with ReLU activation and 512 hidden units
# x = layers.Dense(512, activation='relu')(x)
#
# # Create output layer with a single node and sigmoid activation
# output = layers.Dense(1, activation='sigmoid')(x)
#
# # Create model:
# # input = input feature map
# # output = input feature map + stacked convolution/maxpooling layers + fully
# # connected layer + sigmoid output layer
# model = Model(img_input, output)

###########################################################################################

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Configure and compile the model
model = Model(img_input, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

###########################################################################################

# Model architecture summary.
model.summary()

# Compile the mode using RMSprop optimization algorithm.
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

###########################################################################################
"""
Data preprocessing. (modified to use data augmentation)
"""

# # All images will be rescaled by 1./255
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# # Flow training images in batches of 20 using train_datagen generator
# train_generator = train_datagen.flow_from_directory(
#     train_dir,  # This is the source directory for training images
#     target_size=(150, 150),  # All images will be resized to 150x150
#     batch_size=20,
#     # Since we use binary_crossentropy loss, we need binary labels
#     class_mode='binary')
#
# # Flow validation images in batches of 20 using test_datagen generator
# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow validation images in batches of 32 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

###########################################################################################
"""
Train the model. (modified for Exercise 2)
"""

# start_time = time.time()
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,  # 2000 images = batch_size * steps
#     epochs=15,
#     validation_data=validation_generator,
#     validation_steps=50,  # 1000 images = batch_size * steps
#     verbose=2)
#
# end_time = time.time()
#
# print("Time taken to train: " + str(end_time - start_time))

# WRITE CODE TO TRAIN THE MODEL ON ALL 2000 IMAGES FOR 30 EPOCHS, AND VALIDATE
# ON ALL 1,000 TEST IMAGES
start_time = time.time()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2)

end_time = time.time()

print("Time taken to train: " + str(end_time - start_time))

###########################################################################################

"""
Visualize Intermediate Representations.
"""

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std() # Some kind of error here.
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size: (i + 1) * size] = x
        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

###########################################################################################
"""
Evaluating Accuracy and Loss for the Model.
"""

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.show()

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

###########################################################################################

###########################################################################################

"""
Exercise 9.3 Questions:
###########################################################################################

Exercise 1:
What’s size of the cats/dogs datasets?

Let's start by downloading our example data, a .zip of 2,000 JPG pictures of cats and dogs, 
and extracting it locally in /tmp.

NOTE: The 2,000 images used in this exercise are excerpted from the "Dogs vs. Cats" dataset available on Kaggle, 
which contains 25,000 images. Here, we use a subset of the full dataset to decrease training time for educational 
purposes.

So, 25,000 images in total for the entire dataset but we are using just 2,000, a subset, to expedite training time.

########################################################
How does the first convnet compare with the one we did in class.

The one in Exercise 1 has 3 max_pooling2d layers while the one in class only has 2 max_pooling2d layers.

The one in Exercise 1 has 9,494,561 total trainable parameters while the one in class only has 93,222 trainable
parameters.

The model summary for Exercise 1 shows a input_layer whereas the model summary for the one in class does not.

The specific numerical differences in terms of nodes, dimensions, etc., can be seen below from the copy/pasted
model summary of the one from Exercise 1 and the one from class.

Convolutional Neural Network for Exercise 1 of ML Practicum:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 148, 148, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 17, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 18496)             0         
_________________________________________________________________
dense (Dense)                (None, 512)               9470464   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 9,494,561
Trainable params: 9,494,561
Non-trainable params: 0
_________________________________________________________________

Process finished with exit code 0

##############################

Convolutional Neural Network for In-Class Lecture:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 11, 11, 64)        18496     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 3, 3, 64)          36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 576)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 64)                36928     
_________________________________________________________________
dense_4 (Dense)              (None, 10)                650       
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
_________________________________________________________________

########################################################

Can you see any interesting patterns in the intermediate representations?

As you go deeper, the layers displays smaller representations.
Some of the representations displayed seems like what you would see after applying a Adobe Photoshop filters.
Some of the representations are visibly of cats while others are very abstract.
The representations seem predominantly a hue of green and yellow with some blue.
There are some completely black/blank squares/representations.
Some of the representations seem to be a dot-by-dot yellow outline of parts of the cat.

These are repeating the same single image and manipulating that representation in different ways?

It makes sense the max_pooling layer would decrease the size of the representation as we learned in class it takes the
max value and discards the rest of the data, thereby reducing dimensions.

This is hierarchical decomposition as we go deeper into the layers.

########################################################

Exercise 2:
What is data augmentation?

Artificially boosting the diversity and number of training examples by performing random transformations to 
existing images to create a set of new variants. (scaling, skewing, rotating, translating, etc.)

########################################################

Report your best results and the hyperparameters you used to get them.

This took forever training on 30 epochs in the online notebook >_<.

###########################

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow validation images in batches of 32 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
    
###########################

start_time = time.time()

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
    verbose=2)

end_time = time.time()

print("Time taken to train: " + str(end_time - start_time))

###########################
Results:

(refer to exercise9.3_training_validation_acc.png in Lab09 directory for accuracy metric)
(refer to exercise9.3_training_validation_loss.png in Lab09 directory for loss metric)


 WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/30
50/50 [==============================] - 5s 92ms/step - loss: 0.7350 - acc: 0.5030
 - 21s - loss: 0.8190 - acc: 0.4995 - val_loss: 0.7350 - val_acc: 0.5030
Epoch 2/30
50/50 [==============================] - 5s 91ms/step - loss: 0.6264 - acc: 0.6580
 - 19s - loss: 0.6872 - acc: 0.5725 - val_loss: 0.6264 - val_acc: 0.6580
Epoch 3/30
50/50 [==============================] - 5s 91ms/step - loss: 0.6079 - acc: 0.6780
 - 19s - loss: 0.6641 - acc: 0.6115 - val_loss: 0.6079 - val_acc: 0.6780
Epoch 4/30
50/50 [==============================] - 4s 90ms/step - loss: 0.6129 - acc: 0.6490
 - 19s - loss: 0.6609 - acc: 0.6365 - val_loss: 0.6129 - val_acc: 0.6490
Epoch 5/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5849 - acc: 0.6610
 - 19s - loss: 0.6311 - acc: 0.6650 - val_loss: 0.5849 - val_acc: 0.6610
Epoch 6/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5700 - acc: 0.7160
 - 19s - loss: 0.6322 - acc: 0.6615 - val_loss: 0.5700 - val_acc: 0.7160
Epoch 7/30
50/50 [==============================] - 4s 90ms/step - loss: 0.6864 - acc: 0.6460
 - 19s - loss: 0.6111 - acc: 0.6780 - val_loss: 0.6864 - val_acc: 0.6460
Epoch 8/30
50/50 [==============================] - 5s 90ms/step - loss: 0.5681 - acc: 0.7020
 - 20s - loss: 0.6130 - acc: 0.6725 - val_loss: 0.5681 - val_acc: 0.7020
Epoch 9/30
50/50 [==============================] - 5s 90ms/step - loss: 0.5388 - acc: 0.7180
 - 19s - loss: 0.5916 - acc: 0.6955 - val_loss: 0.5388 - val_acc: 0.7180
Epoch 10/30
50/50 [==============================] - 5s 92ms/step - loss: 0.6784 - acc: 0.6440
 - 19s - loss: 0.5846 - acc: 0.7030 - val_loss: 0.6784 - val_acc: 0.6440
Epoch 11/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5534 - acc: 0.7020
 - 19s - loss: 0.5906 - acc: 0.6840 - val_loss: 0.5534 - val_acc: 0.7020
Epoch 12/30
50/50 [==============================] - 4s 88ms/step - loss: 0.5248 - acc: 0.7420
 - 19s - loss: 0.5829 - acc: 0.7010 - val_loss: 0.5248 - val_acc: 0.7420
Epoch 13/30
50/50 [==============================] - 5s 92ms/step - loss: 0.5198 - acc: 0.7410
 - 19s - loss: 0.5919 - acc: 0.6920 - val_loss: 0.5198 - val_acc: 0.7410
Epoch 14/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5471 - acc: 0.7180
 - 19s - loss: 0.5717 - acc: 0.7090 - val_loss: 0.5471 - val_acc: 0.7180
Epoch 15/30
50/50 [==============================] - 4s 89ms/step - loss: 0.5114 - acc: 0.7390
 - 19s - loss: 0.5682 - acc: 0.7140 - val_loss: 0.5114 - val_acc: 0.7390
Epoch 16/30
50/50 [==============================] - 4s 90ms/step - loss: 0.5501 - acc: 0.7310
 - 19s - loss: 0.5720 - acc: 0.7205 - val_loss: 0.5501 - val_acc: 0.7310
Epoch 17/30
50/50 [==============================] - 4s 90ms/step - loss: 0.5267 - acc: 0.7340
 - 19s - loss: 0.5753 - acc: 0.7155 - val_loss: 0.5267 - val_acc: 0.7340
Epoch 18/30
50/50 [==============================] - 4s 88ms/step - loss: 0.5391 - acc: 0.7200
 - 19s - loss: 0.5593 - acc: 0.7105 - val_loss: 0.5391 - val_acc: 0.7200
Epoch 19/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5033 - acc: 0.7640
 - 19s - loss: 0.5599 - acc: 0.7145 - val_loss: 0.5033 - val_acc: 0.7640
Epoch 20/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5317 - acc: 0.7340
 - 19s - loss: 0.5560 - acc: 0.7280 - val_loss: 0.5317 - val_acc: 0.7340
Epoch 21/30
50/50 [==============================] - 4s 88ms/step - loss: 0.4983 - acc: 0.7410
 - 19s - loss: 0.5315 - acc: 0.7400 - val_loss: 0.4983 - val_acc: 0.7410
Epoch 22/30
50/50 [==============================] - 4s 90ms/step - loss: 0.4957 - acc: 0.7610
 - 19s - loss: 0.5515 - acc: 0.7295 - val_loss: 0.4957 - val_acc: 0.7610
Epoch 23/30
50/50 [==============================] - 5s 91ms/step - loss: 0.5362 - acc: 0.7430
 - 19s - loss: 0.5263 - acc: 0.7565 - val_loss: 0.5362 - val_acc: 0.7430
Epoch 24/30
50/50 [==============================] - 5s 94ms/step - loss: 0.4965 - acc: 0.7440
 - 19s - loss: 0.5482 - acc: 0.7220 - val_loss: 0.4965 - val_acc: 0.7440
Epoch 25/30
50/50 [==============================] - 5s 93ms/step - loss: 0.4796 - acc: 0.7660
 - 19s - loss: 0.5335 - acc: 0.7365 - val_loss: 0.4796 - val_acc: 0.7660
Epoch 26/30
50/50 [==============================] - 4s 90ms/step - loss: 0.5304 - acc: 0.7420
 - 19s - loss: 0.5428 - acc: 0.7255 - val_loss: 0.5304 - val_acc: 0.7420
Epoch 27/30
50/50 [==============================] - 5s 92ms/step - loss: 0.5094 - acc: 0.7490
 - 19s - loss: 0.5341 - acc: 0.7365 - val_loss: 0.5094 - val_acc: 0.7490
Epoch 28/30
50/50 [==============================] - 5s 90ms/step - loss: 0.4787 - acc: 0.7600
 - 19s - loss: 0.5270 - acc: 0.7360 - val_loss: 0.4787 - val_acc: 0.7600
Epoch 29/30
50/50 [==============================] - 4s 90ms/step - loss: 0.4883 - acc: 0.7730
 - 19s - loss: 0.5153 - acc: 0.7445 - val_loss: 0.4883 - val_acc: 0.7730
Epoch 30/30
50/50 [==============================] - 5s 91ms/step - loss: 0.4930 - acc: 0.7690
 - 19s - loss: 0.5193 - acc: 0.7440 - val_loss: 0.4930 - val_acc: 0.7690

 
########################################################

You can skip Exercise 3.

^_^

########################################################
Save your answers in lab09_3.txt.
"""
