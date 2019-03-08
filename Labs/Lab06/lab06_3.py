"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-4-19

Lab 06 - Machine Learning - Numpy

##################################################################################################

Notes:

Exercise 6.3

URL: https://www.tensorflow.org/guide/tensors
(WTF is a tensor?)

URL: https://www.pluralsight.com/guides/different-ways-create-numpy-arrays
(create n-d arrays)

Rank	Math entity
0	Scalar (magnitude only)
1	Vector (magnitude and direction)
2	Matrix (table of numbers)
3	3-Tensor (cube of numbers)
n	n-Tensor (you get the idea)

##################################################################################################

Do the following things using NumPy:

Build samples of the following data structures:

a scalar
a 1D tensor
a 2D tensor
a higher-dimensional tensor

Demonstrate your data structures by printing the rank (i.e., number of dimensions), shape and
data type of each structure.

##################################################################################################

Load the Keras version of the Boston Housing Price dataset (boston_housing) and do the following:

Print the number of training and testing examples.
Print the rank, shape and data type of the examples.
Save your code in lab06_3.py, including all partner’s names in the file.

"""

from keras.datasets import boston_housing

import numpy as np


#######################################################################################################################
#######################################################################################################################


def scalar():
    """
    Creates a scalar data structure.
    :return: Nothing.
    """

    scalar = np.array(3.14)

    print()
    print(str(scalar))
    print("Rank of scalar: " + str(scalar.ndim))
    print("Shape of scalar: " + str(scalar.shape))
    print("Data type of scalar: " + str(scalar.dtype.name))


def tensor_1d():
    """
    Creates a 1-d tensor data structure.
    :return: Nothing.
    """

    tensor_1dimen = np.array([1, 2, 3])

    print()
    print(str(tensor_1dimen))
    print("Rank of 1-d tensor: " + str(tensor_1dimen.ndim))
    print("Shape of 1-d tensor: " + str(tensor_1dimen.shape))
    print("Data type of 1-d tensor: " + str(tensor_1dimen.dtype.name))


def tensor_2d():
    """
    Creates a 2-d tensor data structure.
    :return: Nothing.
    """

    tensor_2dimen = np.array([[1, 2, 3], [4, 5, 6]])

    print()
    print(str(tensor_2dimen))
    print("Rank of 2-d tensor: " + str(tensor_2dimen.ndim))
    print("Shape of 2-d tensor: " + str(tensor_2dimen.shape))
    print("Data type of 2-d tensor: " + str(tensor_2dimen.dtype.name))


def tensor_nd():
    """
    Creates a n-d tensor data structure.
    :return: Nothing.
    """

    tensor_ndimen = np.arange(81).reshape(3, 3, 3, 3)

    print()
    print(str(tensor_ndimen))
    print("Rank of n-d tensor: " + str(tensor_ndimen.ndim))
    print("Shape of n-d tensor: " + str(tensor_ndimen.shape))
    print("Data type of n-d tensor: " + str(tensor_ndimen.dtype.name))


def print_structures():
    """
    Define a function to print the relevant information to console.
    :return: Nothing.
    """
    print(
        f'training images \
            \n\tcount: {len(train_images)} \
            \n\tdimensions: {train_images.ndim} \
            \n\tshape: {train_images.shape} \
            \n\tdata type: {train_images.dtype}\n\n',
        f'testing images \
            \n\tcount: {len(test_labels)} \
            \n\tdimensions: {train_labels.ndim} \
            \n\tshape: {test_labels.shape} \
            \n\tdata type: {test_labels.dtype} \
            \n\tvalues: {test_labels}\n',
    )


#######################################################################################################################


if __name__ == '__main__':
    """
    Main function.
    Executes the program.
    """

    scalar()
    tensor_1d()
    tensor_2d()
    tensor_nd()

    (train_images, train_labels), (test_images, test_labels) = boston_housing.load_data()

    # Print everything to console.
    print_structures()

#######################################################################################################################
