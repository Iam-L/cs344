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

Resources Used:

https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item
(I used this to figure out how to count the number of occurrences of each token.)

https://stackoverflow.com/questions/11068986/how-to-convert-counter-object-to-dict
(convert a "Counter" to a dict

https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
(how to iterate through all items in a dictionary)

https://stackoverflow.com/questions/3845362/how-can-i-check-if-a-key-exists-in-a-dictionary
(how to check if a key exists or doesn't exist in a dictionary)

https://www.pythoncentral.io/how-to-sort-python-dictionaries-by-key-or-value/
(sort a dictionary by its values)
"""

############################################################################################
############################################################################################

from collections import Counter

############################################################################################
############################################################################################

# Define list of words to test.
spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]

# Number of spam messages.
nbad = 5
# Number of non-spam messages.
ngood = 10

"""
Here we go...
"""


# class SpamFilter:
#     """
#     The SpamFilter class accepts a corpus and analyzes it to determine what words are spam and what words are
#     legitimate.
#
#     Parameters:
#         corpus - array of lists containing the words to analyze
#
#     Output:
#         The result of the analysis.
#     """
#
#     def __init__(self, corpus, threshold):
#         """
#         Class constructor.
#
#         :param corpus: list of words to analyze.
#         :param threshold:  threshold value for the algorithm.
#         """
#         self.corpus = corpus
#         self.threshold = threshold
#
#     def wordOccurences(self):
#         """
#         Counts the number of times each word occurs.
#         :param corpus: list of words.
#         :return:  dictionary of each word and their occurrences.
#         """
#         occur_array = []
#
#         for e in self.corpus:
#             occur = Counter(e)
#             occur_array.append(occur)
#
#         return occur_array
#
#     def getWordGoodValue(self):
#         """
#         Gets the specified word's "good" value.
#         :return:  good value.
#         """
#         return 5
#
#     def getWordBadValue(self):
#         """
#         Gets the specified word's "bad" value.
#         :return: bad value.
#         """
#         return 2
#
#     def spam_algorithm(self):
#         """
#         Determines the spam'liness of the word.
#         :return:  spam'liness value.
#         """
#
#         if self.getWordGoodValue() is None:
#             good = 0
#         else:
#             good = 2 * self.getWordGoodValue()
#
#         if self.getWordBadValue() is None:
#             bad = 0
#         else:
#             bad = self.getWordBadValue()
#
#         if good + bad > self.threshold:
#             return max(0.01, min(0.99, min(1.0, bad / nbad) / min(1.0, good / ngood) + min(1.0, bad / nbad)))
#         else:
#             return 0


############################################################################################
############################################################################################

def wordOccurences(corpus):
    """
    Counts the number of times each word occurs.
    :param corpus: list of words.
    :return:  dictionary of each word and their occurrences.
    """
    occur_array = []

    for e in corpus:
        occur = Counter(e)
        occur_array.append(occur)

    return occur_array


def getWordGoodValue():
    """
    Gets the specified word's "good" value.
    :return:  good value.
    """
    return 5


def getWordBadValue():
    """
    Gets the specified word's "bad" value.
    :return: bad value.
    """
    return 2


def spam_algorithm(spam_words, non_spam_words, threshold):
    """
    Determines the spam'liness of the word.
    :return:  spam'liness value.
    """

    # Store calculated probability each word is spam.
    spam_chance = {}

    # Iterate through all non-spam words.
    for good_key, good_value in non_spam_words.items():

        if good_value is None:
            good = 0
        else:
            good = 2 * good_value

        if good_key not in spamWords:
            bad = 0
        else:
            bad = spam_words[good_key]

        if good + bad > threshold:
            probability = max(0.01, min(0.99, min(1.0, bad / nbad) / min(1.0, good / ngood) + min(1.0, bad / nbad)))
        else:
            probability = 0

        # Store to dictionary each word and their associated probability.
        spam_chance[good_key] = probability

    # Return our dictionary of stored word spam probabilities.
    return spam_chance


############################################################################################

if __name__ == '__main__':
    """
    Pithy Introduction.
    """
    print("\nExecuting spam filter algorithm!")
    print("I like Spam! - Delicious!")
    print("\n\n")

    # Threshold value for the spam filter algorithm.
    threshold_value = 1

    # Get occurrences of each word in the list of words.
    tokenDict = wordOccurences(spam_corpus)

    print("Result: " + str(tokenDict))

    # Convert "Counter" to dict.
    spamWords = dict(tokenDict[0])
    nonSpamWords = dict(tokenDict[1])

    print("spam dictionary: " + str(spamWords))
    print("non-spam dictionary: " + str(nonSpamWords))

    word_spam_chance = spam_algorithm(spamWords, nonSpamWords, threshold_value)

    print("word spam probabilities: " + str(word_spam_chance))

    # TODO - figure out how to prune to 15 tokens.
    # If more than 15 tokens, prune to the most "interesting" 15.
    # if len(word_spam_chance) > 1:
    #     sortedDict = sorted(word_spam_chance.items())
    #     print("my sorted dict: " + str(sortedDict))
    #
    #     first15 = islice(sortedDict, 2)
    #     print("my reduced dict: " + str(first15))

    values_only = sorted(word_spam_chance.values())
    print("values only: " + str(values_only))

    # Calculate the product of all probabilities.
    combined_value = 1.0
    for each_one in values_only:
        combined_value *= each_one

    print("product of values: " + str(combined_value))

    # Calculate the complement probabilities.
    complement_values = []

    for each in values_only:
        complement = 1.00 - each
        complement_values.append(complement)

    print("complement values: " + str(complement_values))

    # Calculate the product of all complement probabilities.
    combined_complement_value = 1.0
    for each_one_one in complement_values:
        combined_complement_value *= each_one_one

    print("product of complement values: " + str(combined_complement_value))

    spam_probability_final = combined_value / (combined_value + combined_complement_value)

    print("final probability of spam: " + str(spam_probability_final))

    if spam_probability_final > 0.9:
        print("Spam!!!!")
    else:
        print("Not spam.")


############################################################################################
############################################################################################

"""
Graham argues that this is a Bayesian approach to SPAM. What makes it Bayesian?


"""
