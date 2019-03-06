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

https://www.tutorialspoint.com/How-to-convert-Python-dictionary-keys-values-to-lowercase
(convert dict keys to lower-case)
"""

############################################################################################
############################################################################################

from collections import Counter
from itertools import islice

############################################################################################

# Spam corpus contains spam e-mails.
spam_corpus = [["I", "am", "spam", "spam", "I", "am"], ["I", "do", "not", "like", "that", "spamiam"]]
# Ham corpus contains non-spam e-mails.
ham_corpus = [["do", "i", "like", "green", "eggs", "and", "ham"], ["i", "do"]]
# Test corpus
test_corpus = [["I", "like", "Starcraft", "2"], ["Protoss", "probe", "spam", "is", "annoying"],
               ["You", "must", "construct", "additional", "pylons"]]

"""
The especially observant will notice that while I consider each corpus to be a single long stream of text for purposes 
of counting occurrences, I use the number of emails in each, rather than their combined length, 
as the divisor in calculating spam probabilities. This adds another slight bias to protect against false positives.
"""
# Number of spam and non-spam e-mails messages (not words).
number_bad_message = len(spam_corpus)
number_good_messages = len(ham_corpus)

# Threshold value for the spam filter algorithm.
algorithm_threshold_value = 1
spam_message_threshold_value = 0.9
interesting_tokens_threshold_value = 15


############################################################################################
############################################################################################


class SpamFilter:
    """
    The SpamFilter class accepts a corpus and analyzes it to determine what words are spam and what words are
    legitimate.

    Note: Placeholder for now.

    TODO - refactor spam filter to be class-based rather than just individual floating functions.
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

    ################################################################################

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


def convert_to_lowercase_words_only(spam_occurrences, nonspam_occurrences):
    """
    Convert all keys to lower-case as words should not be case-sensitive.
    Also, collapse all messages and their individual words into one combined dictionary containing
    all unique component words and their occurrences.

    :param spam_occurrences: contains all spam words and their occurrences as multiple dictionaries.
    :param nonspam_occurrences: contains all non-spam words and their occurrences as multiple dictionaries.
    :return:  two dictionaries, one for spam and one for non-spam, containing their lower-case words and
    associated occurrences.

    """
    lowercase_only_spam_word_occurrences = {}
    for each_dictionary in spam_occurrences:
        for key, value in each_dictionary.items():
            if key.lower() not in lowercase_only_spam_word_occurrences:
                lowercase_only_spam_word_occurrences[key.lower()] = value
            else:
                lowercase_only_spam_word_occurrences[key.lower()] += value
    print("\nlower-case only word spam word occurrences: " + str(lowercase_only_spam_word_occurrences))

    lowercase_only_nonspam_word_occurrences = {}
    for each_dictionary in nonspam_occurrences:
        for key, value in each_dictionary.items():
            if key.lower() not in lowercase_only_nonspam_word_occurrences:
                lowercase_only_nonspam_word_occurrences[key.lower()] = value
            else:
                lowercase_only_nonspam_word_occurrences[key.lower()] += value
    print("lower-case only word non-spam word occurrences: " + str(lowercase_only_nonspam_word_occurrences))

    all_occurrences = [lowercase_only_spam_word_occurrences, lowercase_only_nonspam_word_occurrences]
    return all_occurrences


############################################################################################

def individual_word_spam_chance(spam_words_dict, non_spam_words_dict, threshold):
    """
    Determines the spam'liness of each individual word in the message.

    :param spam_words_dict: dictionary containing spam words.
    :param non_spam_words_dict: dictionary containing non-spam words.
    :param threshold: threshold value for statistical algorithm.
    :return:  spam'liness value of each individual word as a dictionary.

    ################################################################################

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

     ################################################################################

    One question that arises in practice is what probability to assign to a word you've never seen, i.e.
    one that doesn't occur in the hash table of word probabilities.
    I've found, again by trial and error, that .4 is a good number to use.
    If you've never seen a word before, it is probably fairly innocent; spam words tend to be all too familiar.

    """

    # Combine unique spam and non-spam words into one dictionary with their associated combined occurrences.
    combined_spam_nonspam_word_occurrences = {}

    for key, value in spam_words_dict.items():
        if key not in combined_spam_nonspam_word_occurrences:
            combined_spam_nonspam_word_occurrences[key] = value
        else:
            combined_spam_nonspam_word_occurrences[key] += value
    print("\nlower-case only spam word occurrences as part of one combined dictionary: "
          + str(combined_spam_nonspam_word_occurrences))

    for key, value in non_spam_words_dict.items():
        if key not in combined_spam_nonspam_word_occurrences:
            combined_spam_nonspam_word_occurrences[key] = value
        else:
            combined_spam_nonspam_word_occurrences[key] += value
    print("lower-case only spam and non-spam word occurrences as part of one combined dictionary: "
          + str(combined_spam_nonspam_word_occurrences))

    ############################################

    # Iterate through all spam and non-spam words in combined dictionary and calculate spam probability for each word.
    words_spam_chance = {}

    for key, value in combined_spam_nonspam_word_occurrences.items():

        # If word is not found in non-spam dictionary set value to 0, otherwise set to value found * 2.
        if key not in non_spam_words_dict:
            good_occurrences = 0
        else:
            good_occurrences = 2 * value

        # If word is not found in spam dictionary set value to 0, otherwise set to that value found.
        if key not in spam_words_dict:
            bad_occurrences = 0
        else:
            bad_occurrences = value

        # Statistical algorithm to calculate the associated probability for each word.
        # Note to self: don't be an idiot and forget a parentheses messing up your order of operations.
        if good_occurrences + bad_occurrences > threshold:
            probability = max(0.01, min(0.99, min(1.0, bad_occurrences / number_bad_message) /
                                        (min(1.0, good_occurrences / number_good_messages) +
                                         min(1.0, bad_occurrences / number_bad_message))))
        else:
            probability = 0.0

        # Store to dictionary each word and their associated probability.
        words_spam_chance[key] = probability

    # Return our dictionary of stored words spam probabilities.
    return words_spam_chance


############################################################################################


def find_interesting_tokens(test_corpus_words, word_spam_chance_dict):
    """
    Prunes dictionary containing the words in the message to the most interesting 15 tokens based
    on the size of their deviation from the "Neutral" value of 0.5

    :param test_corpus_words: the words in the message we wish to determine if it is spam.
    :param word_spam_chance_dict: dictionary containing the spam probabilities of each word.
    :return: the 15 most interesting words and their associated spam probabilities.

    ################################################################################

    When new mail arrives, it is scanned into tokens, and the most interesting fifteen tokens,
    where interesting is measured by how far their spam probability is from a neutral .5,
    are used to calculate the probability that the mail is spam.

    TODO - ask Professor VanderLinden if this is done correctly. (do we use normalized or unormalized values?
    """

    # Assign probabilities to words in the test corpus based on established dataset of words and spam probabilities.
    test_corpus_word_probability_dict = {}
    for each_word in test_corpus_words:
        # If dataset contains that word, assign previously calculated spam probability.
        if each_word.lower() in word_spam_chance_dict.keys():
            test_corpus_word_probability_dict[each_word.lower()] = word_spam_chance_dict[each_word.lower()]
        # If dataset doesn't contain that word, assign specified spam probability value.
        else:
            test_corpus_word_probability_dict[each_word.lower()] = 0.4
    print("\ncontents of test corpus word probability dictionary: " + str(test_corpus_word_probability_dict))

    # If more than 15 tokens, prune to the most "interesting" 15.
    # Determine the 15 tokens with the largest deviation from neutral 0.5.
    normalized_word_spam_chance = {}
    for key, value in test_corpus_word_probability_dict.items():

        # Prevent normalized values = 0.0.
        if value == 0.5:
            normalized_word_spam_chance[key] = abs(0.51 - value)
        else:
            normalized_word_spam_chance[key] = abs(0.5 - value)
    print("\nnormalized word spam chances: " + str(normalized_word_spam_chance))

    # Sort dictionary so that largest deviations are at the front.
    # FIXME - figure out a way to properly sort the dictionary based on value.
    sorted_dict = sorted(normalized_word_spam_chance.items())
    print("\nmy sorted dict with normalized values: " + str(sorted_dict))

    # Slice dictionary so only first 15 key-value pairs are left.
    slice_dict = islice(sorted_dict, interesting_tokens_threshold_value)

    # Convert to dictionary as islice returns an iterator.
    first15 = {}
    for each in slice_dict:
        first15[each[0]] = each[1]
    print("\nfirst 15 tokens with normalized keys and values: " + str(first15))

    # # Un-normalize and return to original values by assigning original values.
    # first15_unnormalized = {}
    # for key, value in first15.items():
    #     first15_unnormalized[key] = word_spam_chance_dict[key]
    # print("first 15 tokens un-normalized keys and values: " + str(first15_unnormalized))

    return first15


############################################################################################

def message_spam_chance(word_probabilities_dict):
    """
    Determines the final probability of the message being spam.

    :param word_probabilities_dict: individual probabilities for each word.
    :return: probability the message is spam.

    ################################################################################

    If probs is a list of the fifteen individual probabilities, you calculate the combined probability thus:

    (let ((prod (apply #'* probs)))
      (/ prod (+ prod (apply #'* (mapcar #'(lambda (x)
                                             (- 1 x))
                                         probs)))))
    """

    # Remove the keys from the dictionary and store only the associated values.
    word_probability_values = sorted(word_probabilities_dict.values())
    print("\nword spam probability values only, keys removed: " + str(word_probability_values))

    # Calculate the product of all individual word probabilities.
    product_of_probabilities = 1.0
    for each_probability in word_probability_values:
        product_of_probabilities *= each_probability
    print("product of individual values: " + str(product_of_probabilities))

    # Determine the complement value of all individual word probabilities.
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
    spamWordOccurrencesDict = word_occurrences(spam_corpus)
    nonSpamWordOccurrencesDict = word_occurrences(ham_corpus)
    print("\noccurrences of each word in spam dictionary: " + str(spamWordOccurrencesDict))
    print("occurrences of each word in non-spam dictionary: " + str(nonSpamWordOccurrencesDict))

    # Convert words in each message to lower-case, sum occurrences of unique words in all messages,
    # and return as list containing 2 dictionaries -one for spam and another for non-spam.
    lower_case_words_only = convert_to_lowercase_words_only(spamWordOccurrencesDict, nonSpamWordOccurrencesDict)

    # Determine probability that each word in the message is spam based on spam and non-spam corpus
    # - returned as dictionary.
    word_spam_chance = individual_word_spam_chance(lower_case_words_only[0],
                                                   lower_case_words_only[1], algorithm_threshold_value)
    print("all words spam probabilities: " + str(word_spam_chance))

    ############################################
    """
    This section is purely to use a test corpus to test for spam'liness against our established dataset.
    """

    # Obtain the 15 most interesting tokens in the message based on their normalized spam probabilities.
    interesting_words_only = find_interesting_tokens(test_corpus[0], word_spam_chance)
    # interesting_words_only = find_interesting_tokens(spam_corpus[0], word_spam_chance)
    # interesting_words_only = find_interesting_tokens(spam_corpus[1], word_spam_chance)
    # interesting_words_only = find_interesting_tokens(ham_corpus[0], word_spam_chance)
    # interesting_words_only = find_interesting_tokens(ham_corpus[1], word_spam_chance)

    # Obtain the spam message probability value.
    result = message_spam_chance(interesting_words_only)

    # Compare spam message probability value against threshold spam probability value.
    print("\nThreshold value above which e-mail message is considered spam: " + str(spam_message_threshold_value))
    if result >= spam_message_threshold_value:
        print("Spam!!!!")
    else:
        print("Not spam.")

############################################################################################
############################################################################################

# TODO - ask Professor VanderLiden about jupyter notebook errors.
