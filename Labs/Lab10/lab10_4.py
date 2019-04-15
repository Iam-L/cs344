"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-31-19
Assignment: Lab 10 - Neural Networks

Notes:

Exercise 10.4 - Intro to Sparse Data and Embeddings

-Included code just to see if I can run it from within PyCharm and because it executes faster.
- Refer to sections below for answers to Exercise questions and code added for Tasks.
"""

###########################################################################################

from __future__ import print_function

import collections
import io
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
train_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)


###########################################################################################

def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        "terms": tf.VarLenFeature(dtype=tf.string),  # terms are strings of varying lengths
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)  # labels are 0 or 1
    }

    parsed_features = tf.parse_single_example(record, features)

    terms = parsed_features['terms'].values
    labels = parsed_features['labels']

    return {'terms': terms}, labels


###########################################################################################

# Create the Dataset object.
ds = tf.data.TFRecordDataset(train_path)
# Map features and labels with the parse function.
ds = ds.map(_parse_function)

# Confirm dataframe creation.
print(ds)
print()

# Retrieve first example in training dataset.
n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
print(sess.run(n))


###########################################################################################

# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, num_epochs=None, shuffle=True):
    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(25, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


###########################################################################################
"""
Task 1: Use a Linear Model with Sparse Inputs and an Explicit Vocabulary
"""

# 50 informative terms that compose our model vocabulary
informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family")

# Create feature column for vocabulary.
terms_feature_column = \
    tf.feature_column.categorical_column_with_vocabulary_list(key="terms", vocabulary_list=informative_terms)

###########################################################################################
"""
Construct Linear Classifier model.
"""

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

feature_columns = [terms_feature_column]

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    optimizer=my_optimizer,
)

# classifier.train(
#   input_fn=lambda: _input_fn([train_path]),
#   steps=1000)
#
# evaluation_metrics = classifier.evaluate(
#   input_fn=lambda: _input_fn([train_path]),
#   steps=1000)
#
# print("Training set metrics:")
# for m in evaluation_metrics:
#   print(m, evaluation_metrics[m])
# print("---")
#
# evaluation_metrics = classifier.evaluate(
#   input_fn=lambda: _input_fn([test_path]),
#   steps=1000)
#
# print("Test set metrics:")
# for m in evaluation_metrics:
#   print(m, evaluation_metrics[m])
# print("---")

###########################################################################################
"""
Task 2: Use a Deep Neural Network (DNN) Model
"""

##################### Here's what we changed ##################################
classifier = tf.estimator.DNNClassifier(  #
    feature_columns=[tf.feature_column.indicator_column(terms_feature_column)],  #
    hidden_units=[20, 20],  #
    optimizer=my_optimizer,  #
)  #
###############################################################################

# try:
#     classifier.train(
#         input_fn=lambda: _input_fn([train_path]),
#         steps=1000)
#
#     evaluation_metrics = classifier.evaluate(
#         input_fn=lambda: _input_fn([train_path]),
#         steps=1)
#     print("Training set metrics:")
#     for m in evaluation_metrics:
#         print(m, evaluation_metrics[m])
#     print("---")
#
#     evaluation_metrics = classifier.evaluate(
#         input_fn=lambda: _input_fn([test_path]),
#         steps=1)
#
#     print("Test set metrics:")
#     for m in evaluation_metrics:
#         print(m, evaluation_metrics[m])
#     print("---")
# except ValueError as err:
#     print(err)

###############################################################################

"""
Task 3: Use an Embedding with a DNN Model

Jacked and modified the code from Exercise 10.3 =)
"""

# Here's a example code snippet you might use to define the feature columns:
terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
feature_columns = [terms_embedding_column]

# Create a DNNClassifier object.
my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    # n_classes=10,
    hidden_units=[20, 20],
    optimizer=my_optimizer,
    config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
)

# classifier.train(
#     input_fn=lambda: _input_fn([train_path]),
#     steps=10)
#
# evaluation_metrics = classifier.evaluate(
#     input_fn=lambda: _input_fn([train_path]),
#     steps=10)
# print("Training set metrics:")
# for m in evaluation_metrics:
#     print(m, evaluation_metrics[m])
# print("---")
#
# evaluation_metrics = classifier.evaluate(
#     input_fn=lambda: _input_fn([test_path]),
#     steps=10)
#
# print("Test set metrics:")
# for m in evaluation_metrics:
#     print(m, evaluation_metrics[m])
# print("---")

###############################################################################
"""
Task 4: Convince yourself there's actually an embedding in there.

NOTE: Remember, in our case, the embedding is a matrix that allows us to project a 
50-dimensional vector down to 2 dimensions.
"""

# # Check to see if we have an embedding layer.
# print(classifier.get_variable_names())
# print()
#
# # Check to see the embedding layer is the correect shape.
# print(classifier.get_variable_value(
#     'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights').shape)
# print()

###############################################################################
"""
Task 5: Examine the Embedding

Run the following code to see the embedding we trained in Task 3. Do things end up where you'd expect?

The negative words appear in Quadrant 2 while the positive words appear in Quadrant 4 with both categories split
along y=x.

Re-train the model by rerunning the code in Task 3, and then run the embedding visualization below again. 
What stays the same? What changes?

Running the training multiple times, we see that the positions of the words do change but positive words stay in the
lower right and negative words stay in the upper left, generally speaking.

Finally, re-train the model again using only 10 steps (which will yield a terrible model). 
Run the embedding visualization below again. What do you see now, and why?

The positive and negative words are now all over the embedding space with no clear division between them.
We are not training long enough for the model to effectively map the embedding space.

"""

import numpy as np
import matplotlib.pyplot as plt

# embedding_matrix = classifier.get_variable_value(
#     'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')
#
# for term_index in range(len(informative_terms)):
#     # Create a one-hot encoding for our term. It has 0s everywhere, except for
#     # a single 1 in the coordinate that corresponds to that term.
#     term_vector = np.zeros(len(informative_terms))
#     term_vector[term_index] = 1
#     # We'll now project that one-hot vector into the embedding space.
#     embedding_xy = np.matmul(term_vector, embedding_matrix)
#     plt.text(embedding_xy[0],
#              embedding_xy[1],
#              informative_terms[term_index])
#
# # Do a little setup to make sure the plot displays nicely.
# plt.rcParams["figure.figsize"] = (15, 15)
# plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
# plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
# plt.show()

###############################################################################
"""
Task 6: Try to improve the model's performance
"""

# Download the vocabulary file.
terms_url = 'https://download.mlcc.google.com/mledu-datasets/sparse-data-embedding/terms.txt'
terms_path = tf.keras.utils.get_file(terms_url.split('/')[-1], terms_url)

# Create a feature column from "terms", using a full vocabulary file.
informative_terms = None
with io.open(terms_path, 'r', encoding='utf8') as f:
    # Convert it to a set first to remove duplicates.
    informative_terms = list(set(f.read().split()))

# terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_list(key="terms",
#                                                                                  vocabulary_list=informative_terms)

# Use the entire vocabulary.
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_file("terms", terms_path)

terms_embedding_column = tf.feature_column.embedding_column(terms_feature_column, dimension=2)
feature_columns = [terms_embedding_column]

my_optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    optimizer=my_optimizer
)

classifier.train(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)

evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn([test_path]),
    steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

"""
Exercise 10.4 Questions:
###########################################################################################

Task 1: Is a linear model ever preferable to a deep NN model?

TODO: review answer to this question.

If you have a linear problem, you could just use a linear model instead of a deep neural network model.
However, if you have a non-linear problem then you shouldn't use a linear model and should consider a deep neural
network instead.

A linear model also generally trains faster due to having fewer parameters to update and layers to back-propagate 
through for the same set of input variables.

So, if fast prototyping or speed is of the essence a linear model would be better.

##################################################

Task 2: Does the NN model do better than the linear model?

Yes, it does based on the accuracy values.  Refer below for Task 1 and Task 2 output.

##################################################

Task 3: Do embeddings do much good for sentiment analysis tasks?

Maybe not, the accuracy metric actually decreased with the addition of a embedding layer.

##################################################

Tasks 4–5: Name two words that have similar embeddings and explain why that makes sense.

"worst" and "terrible" are both adjectives with extreme negative connotations so it makes sense that they would end
up relatively close to each other.

##################################################

Task 6: Report your best hyper-parameters and their resulting performance.

# Use the entire vocabulary.
terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_file("terms", terms_path)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    optimizer=my_optimizer
)

classifier.train(
    input_fn=lambda: _input_fn([train_path]),
    steps=1000)
    
Training set metrics:
accuracy 0.95344
accuracy_baseline 0.5
auc 0.98855644
auc_precision_recall 0.98885506
average_loss 0.13983108
label/mean 0.5
loss 3.495777
precision 0.9466509
prediction/mean 0.50575733
recall 0.96104
global_step 1000
---
Test set metrics:
accuracy 0.87796
accuracy_baseline 0.5
auc 0.9453647
auc_precision_recall 0.94306207
average_loss 0.32169428
label/mean 0.5
loss 8.0423565
precision 0.87714535
prediction/mean 0.4949487
recall 0.87904
global_step 1000
---

Process finished with exit code 0

##################################################

Optional Discussion: You can skip this section.

^_^

##################################################

Task 1 Output:

Training set metrics:
accuracy 0.78944
accuracy_baseline 0.5
auc 0.87196994
auc_precision_recall 0.86309046
average_loss 0.45069042
label/mean 0.5
loss 11.267261
precision 0.76198405
prediction/mean 0.505841
recall 0.84184
global_step 1000
---
Test set metrics:
accuracy 0.78504
accuracy_baseline 0.5
auc 0.8699588
auc_precision_recall 0.8610152
average_loss 0.45197105
label/mean 0.5
loss 11.299276
precision 0.7581884
prediction/mean 0.504314
recall 0.83704
global_step 1000
---

Process finished with exit code 0

##########################

Task 2 Output:

Training set metrics:
accuracy 0.8
accuracy_baseline 0.64
auc 0.86805546
auc_precision_recall 0.8319037
average_loss 0.45062447
label/mean 0.36
loss 11.265612
precision 0.6666667
prediction/mean 0.45283017
recall 0.8888889
global_step 1000
---
Test set metrics:
accuracy 0.8
accuracy_baseline 0.68
auc 0.9044117
auc_precision_recall 0.8889296
average_loss 0.39988434
label/mean 0.32
loss 9.997108
precision 0.6666667
prediction/mean 0.40875465
recall 0.75
global_step 1000
---

Process finished with exit code 0

##########################

Task 3 Output:

Training set metrics:
accuracy 0.78596
accuracy_baseline 0.5
auc 0.8666936
auc_precision_recall 0.8550984
average_loss 0.4560195
label/mean 0.5
loss 11.400487
precision 0.7728834
prediction/mean 0.4929062
recall 0.80992
global_step 1000
---
Test set metrics:
accuracy 0.78228
accuracy_baseline 0.5
auc 0.86569464
auc_precision_recall 0.854914
average_loss 0.4573059
label/mean 0.5
loss 11.432648
precision 0.77186227
prediction/mean 0.49176133
recall 0.80144
global_step 1000
---

Process finished with exit code 0

##########################

Task 4 Output:

['dnn/hiddenlayer_0/bias', 'dnn/hiddenlayer_0/bias/t_0/Adagrad', 'dnn/hiddenlayer_0/kernel', 
'dnn/hiddenlayer_0/kernel/t_0/Adagrad', 'dnn/hiddenlayer_1/bias', 'dnn/hiddenlayer_1/bias/t_0/Adagrad', 
'dnn/hiddenlayer_1/kernel', 'dnn/hiddenlayer_1/kernel/t_0/Adagrad', 
'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights', 
'dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights/t_0/Adagrad', 'dnn/logits/bias', 
'dnn/logits/bias/t_0/Adagrad', 'dnn/logits/kernel', 'dnn/logits/kernel/t_0/Adagrad', 'global_step']

(50, 2)

##########################

Task 5 Output:

Refer to the screen captures in Lab10 directory folder.

##########################

Task 6 Output:

Training set metrics:
accuracy 0.95344
accuracy_baseline 0.5
auc 0.98855644
auc_precision_recall 0.98885506
average_loss 0.13983108
label/mean 0.5
loss 3.495777
precision 0.9466509
prediction/mean 0.50575733
recall 0.96104
global_step 1000
---
Test set metrics:
accuracy 0.87796
accuracy_baseline 0.5
auc 0.9453647
auc_precision_recall 0.94306207
average_loss 0.32169428
label/mean 0.5
loss 8.0423565
precision 0.87714535
prediction/mean 0.4949487
recall 0.87904
global_step 1000
---

Process finished with exit code 0

"""
