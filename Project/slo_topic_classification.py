"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO Topic Classification

###########################################################
Notes:

Proceeding with provided labeled SLO TBL dataset.  Will attempt to preprocess and train this.

Using the "NLTK" Natural Language Toolkit as replacement for CMU Tweet Tagger preprocessor.

Using a combination of Sci-kit Learn, Numpy/Pandas, Tensorflow/Keras, and matplotlib.

###########################################################
Resources Used:

https://stackoverflow.com/questions/13413590/how-to-drop-rows-of-pandas-dataframe-whose-value-in-certain-columns-is-nan
(drop rows with NaN values in columns)

https://www.geeksforgeeks.org/different-ways-to-iterate-over-rows-in-pandas-dataframe/
(pandas row iteration methods)

https://stackoverflow.com/questions/40408471/select-data-when-specific-columns-have-null-value-in-pandas
(create boolean indexing mask)

https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
(drop pandas columns)

https://stackoverflow.com/questions/12850345/how-to-combine-two-data-frames-in-python-pandas
(combine dataframes)

https://stackoverflow.com/questions/23667369/drop-all-duplicate-rows-in-python-pandas
(drop duplicate examples)

https://www.nltk.org/
http://www.nltk.org/book/
(text pre-processing)

https://stackoverflow.com/questions/34784004/python-text-processing-nltk-and-pandas
https://stackoverflow.com/questions/48049087/nltk-based-text-processing-with-pandas
https://stackoverflow.com/questions/44173624/how-to-apply-nltk-word-tokenize-library-on-a-pandas-dataframe-for-twitter-data
(tokenize tweets using pands and nltk)

https://www.dataquest.io/blog/settingwithcopywarning/
(SettingWithCopyWarning explanation)

https://stackoverflow.com/questions/42750551/converting-a-string-to-a-lower-case-pandas
(down-case all text)

https://stackoverflow.com/questions/20490274/how-to-reset-index-in-a-pandas-data-frame
(reindex the dataframe)

###########################################################
Regular expressions section:

https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python/40823105#40823105
(remove URL's)

https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
(remove mentions)

https://www.machinelearningplus.com/python/python-regex-tutorial-examples/
(remove stuff from tweets via regex)

https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
(remove punctuation from strings)

https://www.w3schools.com/python/python_regex.asp
(basic tutorial on regular expressions)

###########################################################
Sci-kit Learn section:

https://www.dataquest.io/blog/sci-kit-learn-tutorial/
(sci-kit learn tutorial)

https://stackoverflow.com/questions/49806790/iterable-over-raw-text-documents-expected-string-object-received
(saved my life)

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
(CounteVectorizer)

https://realpython.com/python-keras-text-classification/
(text classification tutorial using python, sci-kit learn, and keras)

https://nlpforhackers.io/keras-intro/
(text classification using keras and NN's)

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
(encode labels from categorical to numerical)

https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
(followed this one initially for text classification)

https://stackoverflow.com/questions/45804133/dimension-mismatch-error-in-countvectorizer-multinomialnb
(only call fit_transform() once to fit to the dataset; afterwards, use transform() only otherwise issues)

"""

################################################################################################################

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk as nltk
from nltk.tokenize import TweetTokenizer
import re

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

debug = True
################################################################################################################

# Import the dataset.
slo_dataset = \
    pd.read_csv("datasets/tbl_training_set.csv", sep=",")

# Shuffle the data randomly.
slo_dataset = slo_dataset.reindex(
    np.random.permutation(slo_dataset.index))

# Rename columns to something that makes sense.
column_names = ['Tweet', 'SLO1', 'SLO2', 'SLO3']

# Generate a Pandas dataframe.
slo_dataframe = pd.DataFrame(slo_dataset)

# Print shape and column names.
print("The shape of our SLO dataframe:")
print(slo_dataframe.shape)
print()
print("The columns of our SLO dataframe:")
print(slo_dataframe.head)
print()

# Assign column names.
slo_dataframe.columns = column_names

################################################################################################################

# Data pre-processing
# TODO - use pre-processing methods indicated in SLO article
# TODO - https://github.com/Calvin-CS/slo-classifiers/tree/feature/keras-nn/stance/data
# TODO - https://github.com/Calvin-CS/slo-classifiers/blob/feature/keras-nn/stance/data/tweet_preprocessor.py
# FIXME - preprocessor will only work on Linux/Mac

# Drop all rows with only NaN in all columns.
slo_dataframe = slo_dataframe.dropna(how='all')
# Drop all rows without at least 2 non NaN values - indicating no SLO TBL classification labels.
slo_dataframe = slo_dataframe.dropna(thresh=2)

print(slo_dataframe.shape)
print()

if debug:
    # Iterate through each row and check we dropped properly.
    print()
    print("Dataframe only with examples that have SLO TBL classification labels:")
    for index in slo_dataframe.index:
        print(slo_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo_dataframe['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe['SLO2'][index]) + '\tSLO3: ' + str(slo_dataframe['SLO3'][index]))
    print("Shape of dataframe with SLO TBL classifications: " + str(slo_dataframe.shape))

#######################################################

# Boolean indexing to select examples with only a single SLO TBL classification.
mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].isna() & slo_dataframe['SLO3'].isna())

# Check that boolean indexing is working.
print()
print("Check that our boolean indexing mask gives only examples with a single SLO TBL classifications:")
print(mask.tail)
print("The shape of our boolean indexing mask:")
print(mask.shape)

# Create new dataframe with only those examples with a single SLO TBL classification.
slo_dataframe_single_classification = slo_dataframe[mask]

# Check that we have created the new dataframe properly.
if debug:
    # Iterate through each row and check that only examples with multiple SLO TBL classifications are left.
    print("Dataframe only with examples that have a single SLO TBL classification label:")
    for index in slo_dataframe_single_classification.index:
        print(slo_dataframe_single_classification['Tweet'][index]
              + '\tSLO1: ' + str(slo_dataframe_single_classification['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe_single_classification['SLO2'][index])
              + '\tSLO3: ' + str(slo_dataframe_single_classification['SLO3'][index]))
    print("Shape of dataframe with a single SLO TBL classification: "
          + str(slo_dataframe_single_classification.shape))

#######################################################

# Drop SLO2 and SLO3 columns as they are just NaN values.
slo_dataframe_single_classification = slo_dataframe_single_classification.drop(columns=['SLO2', 'SLO3'])

if debug:
    print('\n')
    print("Dataframe with SLOW2 and SLO3 columns dropped as they are just NaN values:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo_dataframe_single_classification.index:
        print(slo_dataframe_single_classification['Tweet'][index] + '\tSLO1: '
              + str(slo_dataframe_single_classification['SLO1'][index]))
    print("Shape of slo_dataframe_single_classification: " + str(slo_dataframe_single_classification.shape))

# Re-name columns.
column_names_single = ['Tweet', 'SLO']

slo_dataframe_single_classification.columns = column_names_single

#######################################################

# Boolean indexing to select examples with multiple SLO TBL classifications.
mask = slo_dataframe['SLO1'].notna() & (slo_dataframe['SLO2'].notna() | slo_dataframe['SLO3'].notna())

# Check that boolean indexing is working.
print()
print("Check that our boolean indexing mask gives only examples with multiple SLO TBL classifications:")
print(mask.tail)
print("The shape of our boolean indexing mask:")
print(mask.shape)

# Create new dataframe with only those examples with multiple SLO TBL classifications.
slo_dataframe_multiple_classifications = slo_dataframe[mask]

# Check that we have created the new dataframe properly.
if debug:
    # Iterate through each row and check that only examples with multiple SLO TBL classifications are left.
    print("Dataframe only with examples that have multiple SLO TBL classification labels:")
    for index in slo_dataframe_multiple_classifications.index:
        print(slo_dataframe_multiple_classifications['Tweet'][index]
              + '\tSLO1: ' + str(slo_dataframe_multiple_classifications['SLO1'][index])
              + '\tSLO2: ' + str(slo_dataframe_multiple_classifications['SLO2'][index])
              + '\tSLO3: ' + str(slo_dataframe_multiple_classifications['SLO3'][index]))
    print("Shape of dataframe with multiple SLO TBL classifications: "
          + str(slo_dataframe_multiple_classifications.shape))

#######################################################

# Duplicate examples with multiple SLO TBL classifications into examples with only 1 SLO TBL classification each.
slo1_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO2', 'SLO3'])
slo2_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO3'])
slo3_dataframe = slo_dataframe_multiple_classifications.drop(columns=['SLO1', 'SLO2'])

if debug:
    print('\n')
    print("Separated dataframes single label for examples with multiple SLO TBL classification labels:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo1_dataframe.index:
        print(slo1_dataframe['Tweet'][index] + '\tSLO1: ' + str(slo1_dataframe['SLO1'][index]))
    for index in slo2_dataframe.index:
        print(slo2_dataframe['Tweet'][index] + '\tSLO2: ' + str(slo2_dataframe['SLO2'][index]))
    for index in slo3_dataframe.index:
        print(slo3_dataframe['Tweet'][index] + '\tSLO3: ' + str(slo3_dataframe['SLO3'][index]))
    print("Shape of slo1_dataframe: " + str(slo1_dataframe.shape))
    print("Shape of slo2_dataframe: " + str(slo2_dataframe.shape))
    print("Shape of slo3_dataframe: " + str(slo3_dataframe.shape))

# Re-name columns.
column_names_single = ['Tweet', 'SLO']

slo1_dataframe.columns = column_names_single
slo2_dataframe.columns = column_names_single
slo3_dataframe.columns = column_names_single

#######################################################

# Concatenate the individual dataframes back together.
frames = [slo1_dataframe, slo2_dataframe, slo3_dataframe, slo_dataframe_single_classification]
slo_dataframe_combined = pd.concat(frames, ignore_index=True)

# Note: Doing this as context-sensitive menu stopped displaying all useable function calls after concat.
slo_dataframe_combined = pd.DataFrame(slo_dataframe_combined)

if debug:
    print('\n')
    print("Recombined individual dataframes for the dataframe representing Tweets with only a single SLO TBL "
          "classification example\n and for the dataframes representing Tweets with multiple SLO TBL classification "
          "labels:")
    # Iterate through each row and check that each example only has one SLO TBL Classification left.
    for index in slo_dataframe_combined.index:
        print(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
    print('Shape of recombined dataframes: ' + str(slo_dataframe_combined.shape))

#######################################################

# Drop all rows with only NaN in all columns.
slo_dataframe_combined = slo_dataframe_combined.dropna()

if debug:
    print('\n')
    print("Recombined dataframes - NaN examples removed:")
    # Iterate through each row and check that we no longer have examples with NaN values.
    for index in slo_dataframe_combined.index:
        print(slo_dataframe_combined['Tweet'][index] + '\tSLO: ' + str(slo_dataframe_combined['SLO'][index]))
    print('Shape of recombined dataframes without NaN examples: ' + str(slo_dataframe_combined.shape))

#######################################################

# Drop duplicate examples with the same SLO TBL classification values.
slo_dataframe_TBL_duplicates_dropped = slo_dataframe_combined.drop_duplicates(subset=['Tweet', 'SLO'], keep=False)

if debug:
    print('\n')
    print("Same examples with duplicate SLO TBL classifications removed:")
    # Iterate through each row and check that we no longer have examples with NaN values.
    for index in slo_dataframe_TBL_duplicates_dropped.index:
        print(slo_dataframe_TBL_duplicates_dropped['Tweet'][index] + '\tSLO: '
              + str(slo_dataframe_TBL_duplicates_dropped['SLO'][index]))
    print('Shape of dataframes without duplicate TBL values: ' + str(slo_dataframe_TBL_duplicates_dropped.shape))


#######################################################

def preprocess_tweets(tweet_text):
    """
    Function performs NLTK text pre-processing.

    Notes:

    Stop words are retained.

    TODO - shrink character elongations
    TODO - remove non-english tweets
    TODO - remove non-company associated tweets
    TODO - remove year and time.
    TODO - remove cash items?

    :return:
    """

    # Remove "RT" tags.
    preprocessed_tweet_text = re.sub("rt", "", tweet_text)

    # Remove URL's.
    preprocessed_tweet_text = re.sub("http[s]?://\S+", "slo_url", preprocessed_tweet_text)

    # Remove Tweet mentions.
    preprocessed_tweet_text = re.sub("@\S+", "slo_mention", preprocessed_tweet_text)

    # Remove Tweet hashtags.
    preprocessed_tweet_text = re.sub("#\S+", "slo_hashtag", preprocessed_tweet_text)

    # Remove all punctuation.
    # preprocessed_tweet_text = preprocessed_tweet_text.translate(str.maketrans('', '', string.punctuation))

    return preprocessed_tweet_text


# Assign new dataframe to contents of old.
slo_df_tokenized = slo_dataframe_TBL_duplicates_dropped

# Down-case all text.
slo_df_tokenized['Tweet'] = slo_df_tokenized['Tweet'].str.lower()

# Pre-process each tweet individually.
for index in slo_df_tokenized.index:
    slo_df_tokenized['Tweet'][index] = preprocess_tweets(slo_df_tokenized['Tweet'][index])

################################################################################################################

# # Use NLTK to tokenize each Tweet.
# tweet_tokenizer = TweetTokenizer()
# slo_df_tokenized['Tweet'] = slo_dataframe_TBL_duplicates_dropped['Tweet'].apply(tweet_tokenizer.tokenize)

# Use for NLTK debugging.
# if debug:
#     print('\n')
#     print("SLO TBL dataframe tokenized:")
#     # Iterate through each row and check that we no longer have examples with NaN values.
#     for index in slo_df_tokenized.index:
#         print(slo_df_tokenized['Tweet'][index])
#     print('Shape of tokenized dataframe: ' + str(slo_df_tokenized.shape))

# for index in slo_df_tokenized.index:
#     slo_df_tokenized['Tweet'][index] = vectorizer.transform(slo_df_tokenized['Tweet'][index]).toarray()

################################################################################################################

# Reindex everything.
slo_df_tokenized.index = pd.RangeIndex(len(slo_df_tokenized.index))
# slo_df_tokenized.index = range(len(slo_df_tokenized.index))

################################################################################################################

# Create input features.
selected_features = slo_df_tokenized[column_names_single]
processed_features = selected_features.copy()

# Check what we are using for input features.
if debug:
    print()
    print("The tweets as a string:")
    print(processed_features['Tweet'])
    print()
    print("SLO TBL classification:")
    print(processed_features['SLO'])

# Create feature and target sets.
slo_feature_input = processed_features['Tweet']
slo_targets = processed_features['SLO']

# Create training and test sets.
from sklearn.model_selection import train_test_split

# Note: these are no longer Pandas format, they're Sci-kit Learn format.
tweet_train, tweet_test, target_train, target_test = train_test_split(slo_feature_input, slo_targets, test_size=0.33,
                                                                      random_state=42)

if debug:
    print("Shape of tweet training set:")
    print(tweet_train.data.shape)
    print("Shape of tweet test set:")
    print(tweet_test.data.shape)
    print("Shape of target training set:")
    print(target_train.data.shape)
    print("Shape of target test set:")
    print(target_test.data.shape)

#######################################################

# Use Sci-kit learn to encode labels into integer values - one assigned integer value per class.
from sklearn import preprocessing

target_label_encoder = preprocessing.LabelEncoder()

target_train_encoded = target_label_encoder.fit_transform(target_train)
target_test_encoded = target_label_encoder.fit_transform(target_test)
target_train_DEcoded = target_label_encoder.inverse_transform(target_train_encoded)
target_test_DEcoded = target_label_encoder.inverse_transform(target_test_encoded)

if debug:
    print("Encoded target training labels:")
    print(target_train_encoded)
    print("Decoded target training labels:")
    print(target_train_DEcoded)

    print("Encoded target test labels:")
    print(target_test_encoded)
    print("Decoded target test labels:")
    print(target_test_DEcoded)

#######################################################

# Use Sci-kit learn to tokenize each Tweet.
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0, lowercase=False)
tweet_train_encoded = vectorizer.fit_transform(tweet_train)
tweet_test_encoded = vectorizer.transform(tweet_test)

if debug:
    print("Vectorized tweet training set:")
    print(tweet_train_encoded)
    print("Vectorized tweet testing set:")
    print(tweet_test_encoded)
    print("Shape of the tweet training set:")
    print(tweet_train_encoded.shape)
    print("Shape of the tweet testing set:")
    print(tweet_test_encoded.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

tweet_train_encoded_tfidf = tfidf_transformer.fit_transform(tweet_train_encoded)
tweet_test_encoded_tfidf = tfidf_transformer.transform(tweet_test_encoded)

if debug:
    print("vectorized tweet training set term frequencies down-sampled:")
    print(tweet_train_encoded_tfidf)
    print("Shape of the tweet training set term frequencies: ")
    print(tweet_train_encoded_tfidf.shape)
    print("vectorized tweet test set term frequencies down-sampled:")
    print(tweet_test_encoded_tfidf)
    print("Shape of the tweet test set term frequencies: ")
    print(tweet_test_encoded_tfidf.shape)

################################################################################################################
"""
Train the model using a variety of different classifiers.
"""

from sklearn.naive_bayes import MultinomialNB

clf_multinomialNB = MultinomialNB().fit(tweet_train_encoded_tfidf, target_train_encoded)

# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score
#
# # create an object of type LinearSVC
# svc_model = LinearSVC(random_state=0)
#
# # train the algorithm on training data and predict using the testing data
# pred = svc_model.fit(tweet_train, target_train).predict(tweet_test)
#
# # print the accuracy score of the model
# print("LinearSVC accuracy : ", accuracy_score(target_test, pred, normalize=True))

################################################################################################################
"""
Make predictions using pre-processed and tokenized Tweets from CMU Tweet Tagger.
Note: This required .csv import and vectorization.

Probably won't be the best generalization to new data as the vocabulary between these two different datasets could
be drastically different.

"""
# Import the dataset.
slo_dataset_cmu = \
    pd.read_csv("borg-SLO classifiers/dataset_20100101-20180510_tok.csv", sep=",")

# Shuffle the data randomly.
slo_dataset_cmu = slo_dataset_cmu.reindex(
    np.random.permutation(slo_dataset_cmu.index))

# Generate a Pandas dataframe.
slo_dataframe_cmu = pd.DataFrame(slo_dataset_cmu)

# Print shape and column names.
print()
print("The shape of our SLO CMU dataframe:")
print(slo_dataframe_cmu.shape)
print()
print("The columns of our SLO CMU dataframe:")
print(slo_dataframe_cmu.head)
print()

# Create input features.
selected_features_cmu = slo_dataframe_cmu['tweet_t']
processed_features_cmu = selected_features_cmu.copy()

# Check what we are using for predictions.
if debug:
    print("The shape of our SLO CMU feature dataframe:")
    print(slo_dataframe_cmu.shape)
    print()
    print("The columns of our SLO CMU feature dataframe:")
    print(processed_features_cmu.head)
    print()

#######################################################

# Vectorize the categorical data for use in predictions.
tweet_predict_encoded = vectorizer.transform(processed_features_cmu)

if debug:
    print("Vectorized tweet predictions set:")
    print(tweet_predict_encoded)
    print("Shape of the tweet predictions set:")
    print(tweet_predict_encoded.shape)
    print()

tweet_predict_encoded_tfidf = tfidf_transformer.transform(tweet_predict_encoded)

if debug:
    print("vectorized tweet predictions set term frequencies down-sampled:")
    print(tweet_predict_encoded_tfidf)
    print("Shape of the tweet predictions set term frequencies: ")
    print(tweet_predict_encoded_tfidf.shape)
    print()

# Generalize to new data and predict.
tweet_generalize_new_data_predictions = clf_multinomialNB.predict(tweet_predict_encoded_tfidf)

# View the results.
# for doc, category in zip(processed_features_cmu, tweet_generalize_new_data_predictions):
#     print('%r => %s' % (doc, category))

################################################################################################################

# Predict using test dataset.
tweet_test_predictions = clf_multinomialNB.predict(tweet_test_encoded_tfidf)

# View the results.
for doc, category in zip(tweet_test, tweet_test_predictions):
    print('%r => %s' % (doc, category))

# Measure accuracy.
print()
print("Accuracy for test set predictions using multinomialNB:")
print(str(np.mean(tweet_test_predictions == target_test_encoded)))

################################################################################################################
"""
Build pipeline for the future.
"""
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('multinomialNB', MultinomialNB()),
])

############################################################################################

"""
Main function.  Execute the program.
"""
# Debug variable.
debug_main = 0

if __name__ == '__main__':
    print()

############################################################################################
