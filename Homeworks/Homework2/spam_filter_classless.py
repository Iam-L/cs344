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
from itertools import islice

############################################################################################
############################################################################################

# Define list of words to test (the message).
spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]

# FIXME - how exactly do we determine the # of good and bad messages? (currently using list lengths inside corpus)
"""
The especially observant will notice that while I consider each corpus to be a single long stream of text for purposes 
of counting occurrences, I use the number of emails in each, rather than their combined length, 
as the divisor in calculating spam probabilities. This adds another slight bias to protect against false positives.
"""
# Number of spam and non-spam messages.
number_bad_message = len(spam_corpus[0])
number_good_messages = len(spam_corpus[1])

# Threshold value for the spam filter algorithm.
algorithm_threshold_value = 1
spam_message_threshold_value = 0.9
interesting_tokens_threshold_value = 15


class SpamFilter:
    """
    The SpamFilter class accepts a corpus and analyzes it to determine what words are spam and what words are
    legitimate.

    # FIXME - refactor spam filter to be class-based rather than just individual floating functions.
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

    Here's a sketch of how I do statistical filtering.
    I start with one corpus of spam and one of nonspam mail.
    At the moment each one has about 4000 messages in it.
    I scan the entire text, including headers and embedded html and javascript, of each message in each corpus.
    I currently consider alphanumeric characters, dashes, apostrophes, and dollar signs to be part of tokens,
    and everything else to be a token separator. (There is probably room for improvement here.)
    I ignore tokens that are all digits, and I also ignore html comments,
    not even considering them as token separators.

    I count the number of times each token (ignoring case, currently) occurs in each corpus.
    At this stage I end up with two large hash tables, one for each corpus, mapping tokens to number of occurrences.
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

        Next I create a third hash table, this time mapping each token to the probability that an email containing
    it is a spam, which I calculate as follows [1]:

    (let ((g (* 2 (or (gethash word good) 0)))
          (b (or (gethash word bad) 0)))
       (unless (< (+ g b) 5)
         (max .01
              (min .99 (float (/ (min 1 (/ b nbad))
                                 (+ (min 1 (/ g ngood))
                                    (min 1 (/ b nbad)))))))))

    where word is the token whose probability we're calculating,
    good and bad are the hash tables I created in the first step,
    and ngood and nbad are the number of nonspam and spam messages respectively.

    FIXME - is this the right way of calculating the probabilities? (refer to passage below from "A Plan for Spam")

    One question that arises in practice is what probability to assign to a word you've never seen, i.e.
    one that doesn't occur in the hash table of word probabilities.
    I've found, again by trial and error, that .4 is a good number to use.
    If you've never seen a word before, it is probably fairly innocent; spam words tend to be all too familiar.
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

    If probs is a list of the fifteen individual probabilities, you calculate the combined probability thus:

    (let ((prod (apply #'* probs)))
      (/ prod (+ prod (apply #'* (mapcar #'(lambda (x)
                                             (- 1 x))
                                         probs)))))
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


def find_interesting_tokens(word_spam_chance_dict):
    """
    Prunes dictionary containing the words in the message to the most interesting 15 tokens based
    on the size of their deviation from the "Neutral" value of 0.5

    :param word_spam_chance_dict: dictionary containing the spam probabilities of each word.
    :return: the 15 most interesting words and their associated spam probabilities.

    FIXME - set cut-off value to 15 after we are finished testing (interesting_tokens_threshold_value).

    When new mail arrives, it is scanned into tokens, and the most interesting fifteen tokens,
    where interesting is measured by how far their spam probability is from a neutral .5,
    are used to calculate the probability that the mail is spam.
    """
    # If more than 15 tokens, prune to the most "interesting" 15.
    if len(word_spam_chance_dict) > interesting_tokens_threshold_value:

        # Determine the 15 tokens with the largest deviation from neutral 0.5.
        normalized_word_spam_chance = {}
        for key, value in word_spam_chance_dict.items():
            normalized_word_spam_chance[key] = abs(0.5 - value)
        print("normalized word spam chances: " + str(normalized_word_spam_chance))

        # Sort dictionary so that largest deviations are at the front.
        sorted_dict = sorted(normalized_word_spam_chance.items())
        print("my sorted dict with normalized values: " + str(sorted_dict))

        # Slice dictionary so only first 15 key-value pairs are left.
        slice_dict = islice(sorted_dict, interesting_tokens_threshold_value)

        # Convert to dictionary as islice returns an iterator.
        first15 = {}
        for each in slice_dict:
            first15[each[0]] = each[1]
        print("first 15 tokens with normalized keys and values: " + str(first15))

        # Un-normalize and return to original values by assigning original values.
        first15_unnormalized = {}
        for key, value in first15.items():
            first15_unnormalized[key] = word_spam_chance_dict[key]
        print("first 15 tokens un-normalized keys and values: " + str(first15_unnormalized))
    else:
        return word_spam_chance_dict

    return first15_unnormalized


############################################################################################

if __name__ == '__main__':
    """
    Pithy Introduction.
    Executes the program.
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

    # Obtain the 15 most interesting tokens in the message.
    interesting_words_only = find_interesting_tokens(word_spam_chance)

    # Obtain the spam message probability value.
    result = message_spam_chance(interesting_words_only)

    # Compare spam message probability value against threshold spam probability value.
    if result >= spam_message_threshold_value:
        print("Spam!!!!")
    else:
        print("Not spam.")

############################################################################################
############################################################################################

"""
Graham argues that this is a Bayesian approach to SPAM. What makes it Bayesian?

-insert answer here.
"""
