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

# Define list of words to test (the message).
spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]

# Number of spam and non-spam messages.
number_bad_message = 5
number_good_messages = 10

# Threshold value for the spam filter algorithm.
algorithm_threshold_value = 1
spam_message_threshold_value = 0.9


class SpamFilter:
    """
    The SpamFilter class accepts a corpus and analyzes it to determine what words are spam and what words are
    legitimate.

    Note: Placeholder for now.
    """

    def __init__(self, corpus, threshold):
        """
        Class constructor.

        :param corpus: list of words to analyze.
        :param threshold:  threshold value for the algorithm.
        """
        self.corpus = corpus
        self.threshold = threshold


############################################################################################
############################################################################################

def word_occurrences(corpus):
    """
    Counts the number of times each word occurs in the message.

    :param corpus: list of words to count occurrences for.
    :return:  dictionary of each word and their occurrences.
    """
    occur_array = []

    for e in corpus:
        occur = Counter(e)
        occur_array.append(occur)

    return occur_array


############################################################################################

def individual_word_spam_chance(spam_words_dict, non_spam_words_dict, threshold):
    """
    TODO - ensure this is how the algorithm is meant to function - ask Professor VanderLinden.
    Determines the spam'liness of each individual word in the message.

    :param spam_words_dict: dictionary containing spam words.
    :param non_spam_words_dict: dictionary containing non-spam words.
    :param threshold: threshold value for statistical algorithm.
    :return:  spam'liness value of each individual word as a dictionary.
    """
    words_spam_chance = {}

    # Iterate through all non-spam words.
    for good_key, good_value in non_spam_words_dict.items():

        # If no associated value for each word set to 0, otherwise multiply value by 2.
        if good_value is None:
            good_occurrences = 0
        else:
            good_occurrences = 2 * good_value

        # Determine if non-spam word has match in spam word dictionary.
        # If no, set to 0.  If yes, set to value of non-spam word.
        if good_key not in spam_words_dict:
            bad_occurrences = 0
        else:
            bad_occurrences = spam_words_dict[good_key]

        # Statistical algorithm to calculate the associated probability for each word.
        if good_occurrences + bad_occurrences > threshold:
            probability = max(0.01, min(0.99, min(1.0, bad_occurrences / number_bad_message) /
                                        min(1.0, good_occurrences / number_good_messages) +
                                        min(1.0, bad_occurrences / number_bad_message)))
        else:
            probability = 0

        # Store to dictionary each word and their associated probability.
        words_spam_chance[good_key] = probability

    # Return our dictionary of stored words spam probabilities.
    return words_spam_chance


############################################################################################

def message_spam_chance(word_probabilities_dict):
    """
    Determines the final probability of being spam.

    :param word_probabilities_dict: individual probabilities for each word.
    :return: probability the message is spam.
    """

    # Remove the keys from the dictionary and store only the associated values.
    word_probability_values = sorted(word_probabilities_dict.values())
    print("word spam probability values only, keys removed: " + str(word_probability_values))

    # Calculate the product of all individual word probabilities.
    product_of_probabilities = 1.0
    for each_probability in word_probability_values:
        product_of_probabilities *= each_probability

    print("product of individual values: " + str(product_of_probabilities))

    # Determine the complement of all individual word probabilities.
    word_probability_complement_values = []
    for each_probability in word_probability_values:
        complement = 1.00 - each_probability
        word_probability_complement_values.append(complement)

    print("word spam complement probability values: " + str(word_probability_complement_values))

    # Calculate the product of all complement probabilities.
    product_of_complement_probabilities = 1.0
    for each_complement_probability in word_probability_complement_values:
        product_of_complement_probabilities *= each_complement_probability

    print("product of complement values: " + str(product_of_complement_probabilities))

    spam_message_probability = product_of_probabilities / \
                               (product_of_probabilities + product_of_complement_probabilities)

    print("final probability message is spam: " + str(spam_message_probability))

    return spam_message_probability


############################################################################################

if __name__ == '__main__':
    """
    Pithy Introduction.
    """
    print("\nExecuting spam filter algorithm!")
    print("I like Spam! - Delicious!")
    print("\n\n")

    # Get occurrences of each word in the list of words - returned as dictionary.
    tokenOccurrencesDict = word_occurrences(spam_corpus)

    print("occurrences of each word in spam and non-spam dictionary: " + str(tokenOccurrencesDict))

    # Convert "Counter" to dict.
    spamWordsDict = dict(tokenOccurrencesDict[0])
    nonSpamWordsDict = dict(tokenOccurrencesDict[1])

    print("spam dictionary: " + str(spamWordsDict))
    print("non-spam dictionary: " + str(nonSpamWordsDict))

    # Determine probability that each word in the message is spam - returned as dictionary.
    word_spam_chance = individual_word_spam_chance(spamWordsDict, nonSpamWordsDict, algorithm_threshold_value)

    print("words spam probabilities: " + str(word_spam_chance))

    # TODO - figure out how to prune to 15 tokens.
    # If more than 15 tokens, prune to the most "interesting" 15.
    # if len(word_spam_chance) > 1:
    #     sortedDict = sorted(word_spam_chance.items())
    #     print("my sorted dict: " + str(sortedDict))
    #
    #     first15 = islice(sortedDict, 2)
    #     print("my reduced dict: " + str(first15))

    result = message_spam_chance(word_spam_chance)

    if result > spam_message_threshold_value:
        print("Spam!!!!")
    else:
        print("Not spam.")

############################################################################################
############################################################################################

"""
Graham argues that this is a Bayesian approach to SPAM. What makes it Bayesian?

-insert answer here.
"""
