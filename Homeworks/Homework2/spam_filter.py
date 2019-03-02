"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 3-1-19

Homework 2 - Uncertainty
Spam filter - LISP to Python algorithm conversion (from Paul Graham's A Plan for Spam)

Notes:

Include in your solution, as one test case, the probability tables for the words in the following hard-coded
SPAM/HAM corpus (and only this corpus) using a minimum count threshold of 1 (rather than the 5 used in the algorithm)

TODO - finish.
FIXME - none so far.

Oh, god, functional languages...

"""

############################################################################################
############################################################################################

import time
import math
import random

############################################################################################
############################################################################################

# Define list of words to test.
spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]

corpus = []

"""
Here we go...
"""


class SpamFilter(corpus):
    """
    The SpamFilter class accepts a corpus and analyzes it to determine what words are spam and what words are
    legitimate.

    Parameters:
        corpus - array of lists containing the words to analyze

    Output:
        The result of the analysis.

    """
    # Stub.
    local = corpus


############################################################################################
############################################################################################

if __name__ == '__main__':
    """
    Pithy Introduction.
    """
    print("\nExecuting spam filter algorithm!")
    print("I like Spam! - Delicious!")
    print("\n\n")

############################################################################################
############################################################################################

"""
Graham argues that this is a Baysian approach to SPAM. What makes it Bayesian?


"""
