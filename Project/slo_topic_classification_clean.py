"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 4-23-19

Final Project - SLO TBL Topic Classification

###########################################################
Notes:

Utilizes Scikit-Learn machine learning algorithms for fast prototyping and topic classification using a variety
of Classifiers.

TODO - resolve SettingWithCopyWarning.

TODO - implement data visualizations via matplotlib and Seaborn.

TODO - attempt to acquire additional labeled Tweets for topic classification using pattern matching and pandas queries.
TODO - reference settings.py and autocoding.py for template of how to do this.

TODO - revise report.ipynb and paper as updates are made to implementation and code-base.

###########################################################
Resources Used:

Refer to original un-cleaned version.

https://scikit-plot.readthedocs.io/en/stable/index.html
(visualizations simplified)

"""

################################################################################################################
################################################################################################################

import logging as log
import warnings
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk as nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn import metrics

#############################################################

# Note: FIXME - indicates unresolved import error, but still runs fine.
# noinspection PyUnresolvedReferences
from SLO_TBL_Tweet_Preprocessor_Specialized import tweet_dataset_preprocessor_1, tweet_dataset_preprocessor_2, \
    tweet_dataset_preprocessor_3

#############################################################

# Note: Need to set level AND turn on debug variables in order to see all debug output.
log.basicConfig(level=log.DEBUG)
tf.logging.set_verbosity(tf.logging.ERROR)

# Miscellaneous parameter adjustments for pandas and python.
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Turn on and off to debug various sub-sections.
debug = False
debug_pipeline = False
debug_preprocess_tweets = False
debug_train_test_set_creation = False
debug_classifier_iterations = False
debug_create_prediction_set = False
debug_make_predictions = False

################################################################################################################
################################################################################################################

# Import the datasets.
tweet_dataset_processed1 = \
    pd.read_csv("preprocessed-datasets/tbl_kvlinden_PROCESSED.csv", sep=",")

tweet_dataset_processed2 = \
    pd.read_csv("preprocessed-datasets/tbl_training_set_PROCESSED.csv", sep=",")

# Reindex and shuffle the data randomly.
tweet_dataset_processed1 = tweet_dataset_processed1.reindex(
    pd.np.random.permutation(tweet_dataset_processed1.index))

tweet_dataset_processed2 = tweet_dataset_processed2.reindex(
    pd.np.random.permutation(tweet_dataset_processed2.index))

# Generate a Pandas dataframe.
tweet_dataframe_processed1 = pd.DataFrame(tweet_dataset_processed1)
tweet_dataframe_processed2 = pd.DataFrame(tweet_dataset_processed2)

if debug_preprocess_tweets:
    # Print shape and column names.
    log.debug("\n")
    log.debug("The shape of our SLO dataframe 1:")
    log.debug(tweet_dataframe_processed1.shape)
    log.debug("\n")
    log.debug("The columns of our SLO dataframe 1:")
    log.debug(tweet_dataframe_processed1.head)
    log.debug("\n")
    # Print shape and column names.
    log.debug("\n")
    log.debug("The shape of our SLO dataframe 2:")
    log.debug(tweet_dataframe_processed2.shape)
    log.debug("\n")
    log.debug("The columns of our SLO dataframe 2:")
    log.debug(tweet_dataframe_processed2.head)
    log.debug("\n")

# Concatenate the individual datasets together.
frames = [tweet_dataframe_processed1, tweet_dataframe_processed2]
slo_dataframe_combined = pd.concat(frames, ignore_index=True)

# Reindex everything.
slo_dataframe_combined.index = pd.RangeIndex(len(slo_dataframe_combined.index))
# slo_dataframe_combined.index = range(len(slo_dataframe_combined.index))

# Assign column names.
tweet_dataframe_processed_column_names = ['Tweet', 'SLO']

# Create input features.
selected_features = slo_dataframe_combined[tweet_dataframe_processed_column_names]
processed_features = selected_features.copy()

if debug_preprocess_tweets:
    # Check what we are using as inputs.
    log.debug("\n")
    log.debug("The Tweets in our input feature:")
    log.debug(processed_features['Tweet'])
    log.debug("\n")
    log.debug("SLO TBL topic classification label for each Tweet:")
    log.debug(processed_features['SLO'])
    log.debug("\n")

# Create feature set and target sets.
slo_feature_set = processed_features['Tweet']
slo_target_set = processed_features['SLO']


#######################################################

def create_training_and_test_set():
    """
    This functions splits the feature and target set into training and test sets for each set.

    Note: We use this to generate a randomized training and target set in order to average our results over
    n iterations.

    random_state = rng (where rng = random number seed generator)

    :return: Nothing.  Global variables are established.
    """
    global tweet_train, tweet_test, target_train, target_test, target_train_encoded, target_test_encoded

    from sklearn.model_selection import train_test_split

    import random
    rng = random.randint(1, 1000000)
    # Split feature and target set into training and test sets for each set.
    tweet_train, tweet_test, target_train, target_test = train_test_split(slo_feature_set, slo_target_set,
                                                                          test_size=0.33,
                                                                          random_state=rng)

    if debug_train_test_set_creation:
        log.debug("Shape of tweet training set:")
        log.debug(tweet_train.data.shape)
        log.debug("Shape of tweet test set:")
        log.debug(tweet_test.data.shape)
        log.debug("Shape of target training set:")
        log.debug(target_train.data.shape)
        log.debug("Shape of target test set:")
        log.debug(target_test.data.shape)
        log.debug("\n")

    #######################################################

    # Use Sci-kit learn to encode labels into integer values - one assigned integer value per class.
    from sklearn import preprocessing

    target_label_encoder = preprocessing.LabelEncoder()
    target_train_encoded = target_label_encoder.fit_transform(target_train)
    target_test_encoded = target_label_encoder.fit_transform(target_test)

    target_train_decoded = target_label_encoder.inverse_transform(target_train_encoded)
    target_test_decoded = target_label_encoder.inverse_transform(target_test_encoded)

    if debug_train_test_set_creation:
        log.debug("Encoded target training labels:")
        log.debug(target_train_encoded)
        log.debug("Decoded target training labels:")
        log.debug(target_train_decoded)
        log.debug("\n")
        log.debug("Encoded target test labels:")
        log.debug(target_test_encoded)
        log.debug("Decoded target test labels:")
        log.debug(target_test_decoded)
        log.debug("\n")

    # return [tweet_train, tweet_test, target_train, target_test, target_train_encoded, target_test_encoded]


#######################################################

def scikit_learn_multinomialnb_classifier_non_pipeline():
    """
    Function trains a Multinomial Naive Bayes Classifier without using a Pipeline.

    Note: Implemented for educational purposes - so I can see the manual workflow, otherwise the Pipeline Class hides
    these details and we only have to tune parameters.

    :return: none.
    """

    # Create the training and test sets from the feature and target sets.
    create_training_and_test_set()

    # Use Sci-kit learn to tokenize each Tweet and convert into a bag-of-words sparse feature vector.
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    tweet_train_encoded = vectorizer.fit_transform(tweet_train)
    tweet_test_encoded = vectorizer.transform(tweet_test)

    if debug:
        log.debug("Vectorized tweet training set:")
        log.debug(tweet_train_encoded)
        log.debug("Vectorized tweet testing set:")
        log.debug(tweet_test_encoded)
        log.debug("Shape of the tweet training set:")
        log.debug(tweet_train_encoded.shape)
        log.debug("Shape of the tweet testing set:")
        log.debug(tweet_test_encoded.shape)

    #######################################################

    # Use Sci-kit learn to convert each tokenized Tweet into term frequencies.
    tfidf_transformer = TfidfTransformer()

    tweet_train_encoded_tfidf = tfidf_transformer.fit_transform(tweet_train_encoded)
    tweet_test_encoded_tfidf = tfidf_transformer.transform(tweet_test_encoded)

    if debug:
        log.debug("vectorized tweet training set term frequencies down-sampled:")
        log.debug(tweet_train_encoded_tfidf)
        log.debug("Shape of the tweet training set term frequencies down-sampled: ")
        log.debug(tweet_train_encoded_tfidf.shape)
        log.debug("\n")
        log.debug("vectorized tweet test set term frequencies down-sampled:")
        log.debug(tweet_test_encoded_tfidf)
        log.debug("Shape of the tweet test set term frequencies down-sampled: ")
        log.debug(tweet_test_encoded_tfidf.shape)
        log.debug("\n")

    #######################################################

    from sklearn.naive_bayes import MultinomialNB

    # Train the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf_multinomial_nb.fit(tweet_train_encoded_tfidf, target_train_encoded)

    # Predict using the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb_predict = clf_multinomial_nb.predict(tweet_test_encoded_tfidf)

    from sklearn.metrics import accuracy_score

    log.debug("MultinomialNB Classifier accuracy using accuracy_score() function : ",
              accuracy_score(target_test_encoded, clf_multinomial_nb_predict, normalize=True))
    log.debug("\n")

    # Another method of obtaining accuracy metric.
    log.debug("Accuracy for test set predictions using multinomialNB:")
    log.debug(str(np.mean(clf_multinomial_nb_predict == target_test_encoded)))
    log.debug("\n")

    # View the results as Tweet => predicted topic classification label.
    for doc, category in zip(tweet_test, clf_multinomial_nb_predict):
        log.debug('%r => %s' % (doc, category))


################################################################################################################
def create_prediction_set():
    """
    Function prepares the borg-classifier dataset to be used for predictions in trained models.

    :return: the prepared dataset.
    """

    # Import the dataset.
    slo_dataset_cmu = \
        pd.read_csv("preprocessed-datasets/dataset_20100101-20180510_tok_PROCESSED.csv", sep=",")

    # Shuffle the data randomly.
    slo_dataset_cmu = slo_dataset_cmu.reindex(
        pd.np.random.permutation(slo_dataset_cmu.index))

    # Generate a Pandas dataframe.
    slo_dataframe_cmu = pd.DataFrame(slo_dataset_cmu)

    if debug_create_prediction_set:
        # Print shape and column names.
        log.debug("\n")
        log.debug("The shape of our SLO CMU dataframe:")
        log.debug(slo_dataframe_cmu.shape)
        log.debug("\n")
        log.debug("The columns of our SLO CMU dataframe:")
        log.debug(slo_dataframe_cmu.head)
        log.debug("\n")

    # Reindex everything.
    slo_dataframe_cmu.index = pd.RangeIndex(len(slo_dataframe_cmu.index))
    # slo_dataframe_cmu.index = range(len(slo_dataframe_cmu.index))

    # Create input features.
    # Note: using "filter()" - other methods seems to result in shape of (658982, ) instead of (658982, 1)
    selected_features_cmu = slo_dataframe_cmu.filter(['tweet_t'])
    processed_features_cmu = selected_features_cmu.copy()

    # Rename column.
    processed_features_cmu.columns = ['Tweets']

    if debug_create_prediction_set:
        # Print shape and column names.
        log.debug("\n")
        log.debug("The shape of our processed features:")
        log.debug(processed_features_cmu.shape)
        log.debug("\n")
        log.debug("The columns of our processed features:")
        log.debug(processed_features_cmu.head)
        log.debug("\n")

    if debug_create_prediction_set:
        # Check what we are using as inputs.
        log.debug("\n")
        log.debug("The Tweets in our input feature:")
        log.debug(processed_features_cmu['Tweets'])
        log.debug("\n")

    return processed_features_cmu


################################################################################################################

def make_predictions(trained_model):
    """
    Function makes predictions using the trained model passed as an argument.

    :param trained_model
    :return: Nothingl.
    """

    # Generate the dataset to be used for predictions.
    prediction_set = create_prediction_set()

    # Make predictions of the borg-slo-classifiers dataset.
    # Note to self: don't be an idiot and try to make predictions on the entire dataframe object instead of a column.
    predictions = trained_model.predict(prediction_set['Tweets'])

    # Store predictions in Pandas dataframe.
    results_df = pd.DataFrame(predictions)

    # Assign column names.
    results_df_column_name = ['TBL_classification']
    results_df.columns = results_df_column_name

    if debug_make_predictions:
        log.debug("The shape of our prediction results dataframe:")
        log.debug(results_df.shape)
        log.debug("\n")
        log.debug("The contents of our prediction results dataframe:")
        log.debug(results_df.head())
        log.debug("\n")

    # Count # of each classifications made.
    social_counter = 0
    economic_counter = 0
    environmental_counter = 0

    for index in results_df.index:
        if results_df['TBL_classification'][index] == 'economic':
            economic_counter += 1
        if results_df['TBL_classification'][index] == 'social':
            social_counter += 1
        if results_df['TBL_classification'][index] == 'environmental':
            environmental_counter += 1

    # Calculate percentages for each classification.
    social_percentage = (social_counter / results_df.shape[0]) * 100.0
    economic_percentage = (economic_counter / results_df.shape[0]) * 100.0
    environmental_percentage = (environmental_counter / results_df.shape[0]) * 100.0

    # Display our statistics.
    log.debug("The number of Tweets identified as social is :" + str(social_counter))
    log.debug("The % of Tweets identified as social in the entire dataset is: " + str(social_percentage))
    log.debug("The number of Tweets identified as economic is :" + str(economic_counter))
    log.debug("The % of Tweets identified as economic in the entire dataset is: " + str(economic_percentage))
    log.debug("The number of Tweets identified as environmental is :" + str(environmental_counter))
    log.debug("The % of Tweets identified as environmental in the entire dataset is: " + str(environmental_percentage))
    log.debug("\n")


################################################################################################################
def multinomial_naive_bayes_classifier_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn.naive_bayes import MultinomialNB

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    multinomial_nb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10],
    }

    # Perform the grid search using all cores.
    multinomial_nb_clf = GridSearchCV(multinomial_nb_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    multinomial_nb_clf.fit(tweet_train, target_train)
    multinomial_nb_predictions = multinomial_nb_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(multinomial_nb_clf.cv_results_)
        log.debug("The shape of the Multinomial Naive Bayes Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the Multinomial Naive Bayes Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Multinomial Naive Bayes Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, multinomial_nb_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Multinomial Naive Bayes Classifier Grid Search is: ")
    log.debug(np.mean(multinomial_nb_predictions == target_test))
    log.debug("\n")


################################################################################################################
def multinomial_naive_bayes_classifier():
    """
    Functions trains a Multinomial Naive Bayes Classifier.

    :return: none.
    """
    from sklearn.naive_bayes import MultinomialNB

    multinomial_nb_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        multinomial_nb_clf.fit(tweet_train, target_train)
        multinomial_nb_predictions = multinomial_nb_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(multinomial_nb_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Multinomial Naive Bayes Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("Multinomial Naive Bayes Classifier Metrics")
            log.debug(metrics.classification_report(target_test, multinomial_nb_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("Multinomial Naive Bayes Classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, multinomial_nb_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Multinomial Naive Bayes Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Multinomial Naive Bayes Classifier:")
    make_predictions(multinomial_nb_clf)


################################################################################################################

def sgd_classifier_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn.linear_model import SGDClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    sgd_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                              l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
                              n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
                              power_t=0.5, random_state=None, shuffle=True, tol=None,
                              validation_fraction=0.1, verbose=0, warm_start=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-1, 1e-2, 1e-3, 0.00001, 0.000001),
    }

    # Perform the grid search using all cores.
    sgd_classifier_clf = GridSearchCV(sgd_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    sgd_classifier_clf.fit(tweet_train, target_train)
    sgd_classifier_predictions = sgd_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(sgd_classifier_clf.cv_results_)
        log.debug("The shape of the SGD Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the SGD Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the SGD Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, sgd_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Stochastic Gradient Descent Classifier Grid Search is: ")
    log.debug(np.mean(sgd_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def sgd_classifier():
    """
    Function trains a Stochastic Gradient Descent Classifier.
    
    :return: none.
    """
    from sklearn.linear_model import SGDClassifier

    sgd_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier(alpha=0.1, average=False, class_weight=None,
                              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                              l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=5,
                              n_iter=None, n_iter_no_change=5, n_jobs=None, penalty='l2',
                              power_t=0.5, random_state=None, shuffle=True, tol=None,
                              validation_fraction=0.1, verbose=0, warm_start=False)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        sgd_classifier_clf.fit(tweet_train, target_train)
        sgd_classifier_predictions = sgd_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(sgd_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Stochastic Gradient Descent Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("SGD_classifier Classifier Metrics")
            log.debug(metrics.classification_report(target_test, sgd_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("SGD_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, sgd_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Stochastic Gradient Descent Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Stochastic Gradient Descent Classifier:")
    make_predictions(sgd_classifier_clf)


################################################################################################################
def svm_support_vector_classification_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn import svm

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'clf__gamma': ['scale', 'auto'],
        'clf__shrinking': (True, False),
        'clf__probability': (True, False),
        'clf__tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'clf__decision_function_shape': ['ovo', 'ovr'],
    }

    # Perform the grid search using all cores.
    svc_classifier_clf = GridSearchCV(svc_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    svc_classifier_clf.fit(tweet_train, target_train)
    svc_classifier_predictions = svc_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(svc_classifier_clf.cv_results_)
        log.debug("The shape of the Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug("The contents of the Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Support Vector Classification Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, svc_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Support Vector Classification Classifier Grid Search is: ")
    log.debug(np.mean(svc_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def svm_support_vector_classification():
    """
    Functions trains a Support Vector Machine - Support Vector Classification Classifier.
    
    :return: none.
    """
    from sklearn import svm

    svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.SVC(C=0.9, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovo', degree=3, gamma='scale', kernel='sigmoid',
                        max_iter=-1, probability=True, random_state=None, shrinking=True,
                        tol=0.01, verbose=False)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        svc_classifier_clf.fit(tweet_train, target_train)
        svc_classifier_predictions = svc_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(svc_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Support Vector Classification Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("SVC_classifier Metrics")
            log.debug(metrics.classification_report(target_test, svc_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("SVC_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, svc_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Support Vector Classification Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Support Vector Classification Classifier:")
    make_predictions(svc_classifier_clf)


################################################################################################################
def svm_linear_support_vector_classification_grid_search():
    """
    Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

    :return: Nothing.
    """
    from sklearn import svm

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    linear_svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.LinearSVC(C=0.7, class_weight=None, dual=True, fit_intercept=True,
                              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                              verbose=0)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'clf__penalty': ['l2'],
        'clf__loss': ['squared_hinge'],
        # 'clf__dual': (True, False),
        'clf__multi_class': ['ovr', 'crammer_singer'],
        'clf__tol': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'clf__fit_intercept': (True, False),
        'clf__max_iter': [500, 1000, 1500, 2000],
    }

    # Perform the grid search using all cores.
    linear_svc_classifier_clf = GridSearchCV(linear_svc_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    linear_svc_classifier_clf.fit(tweet_train, target_train)
    linear_svc_classifier_predictions = linear_svc_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(linear_svc_classifier_clf.cv_results_)
        log.debug("The shape of the Linear Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Linear Support Vector Classification Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Linear Support Vector Classification Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, linear_svc_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Linear Support Vector Classification Classifier Grid Search is: ")
    log.debug(np.mean(linear_svc_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def svm_linear_support_vector_classification():
    """"
    Function trains a Support Vector Machine - Linear Support Vector Classification Classifier.
    
    :return: none.
    """
    from sklearn import svm

    linear_svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', svm.LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
                              intercept_scaling=1, loss='squared_hinge', max_iter=2000,
                              multi_class='ovr', penalty='l2', random_state=None, tol=0.1,
                              verbose=0)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        linear_svc_classifier_clf.fit(tweet_train, target_train)
        linear_svc_classifier_predictions = linear_svc_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(linear_svc_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Linear Support Vector Classification Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("LinearSVC_classifier Metrics")
            log.debug(metrics.classification_report(target_test, linear_svc_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("LinearSVC_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, linear_svc_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Linear Support Vector Classification Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Linear Support Vector Classification Classifier:")
    make_predictions(linear_svc_classifier_clf)


################################################################################################################
def nearest_kneighbor_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn.neighbors import KNeighborsClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    k_neighbor_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__n_neighbors': [10, 15, 20, 25, 30],
        'clf__weights': ['uniform', 'distance'],
        'clf__algorithm': ['auto'],
        'clf__leaf_size': [5, 10, 15, 20],
        'clf__p': [1, 2, 3, 4],
        'clf__metric': ['euclidean', 'manhattan'],
    }

    # Perform the grid search using all cores.
    k_neighbor_classifier_clf = GridSearchCV(k_neighbor_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    k_neighbor_classifier_clf.fit(tweet_train, target_train)
    k_neighbor_classifier_predictions = k_neighbor_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(k_neighbor_classifier_clf.cv_results_)
        log.debug("The shape of the KNeighbor Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the  KNeighbor Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the KNeighbor Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, k_neighbor_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using KNeighbor Classifier Grid Search is: ")
    log.debug(np.mean(k_neighbor_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def nearest_kneighbor_classifier():
    """
    Function trains a Nearest Neighbor - KNeighbor Classifier.
    
    :return: none. 
    """
    from sklearn.neighbors import KNeighborsClassifier

    k_neighbor_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', KNeighborsClassifier(n_neighbors=30, algorithm='auto', leaf_size=10, metric='euclidean', p=1,
                                     weights='uniform', n_jobs=-1)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        k_neighbor_classifier_clf.fit(tweet_train, target_train)
        k_neighbor_classifier_predictions = k_neighbor_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(k_neighbor_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using KNeighbor Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("KNeighbor_classifier Metrics")
            log.debug(metrics.classification_report(target_test, k_neighbor_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("KNeighbor_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, k_neighbor_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("KNeighbor Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using KNeighbor Classifier:")
    make_predictions(k_neighbor_classifier_clf)


################################################################################################################
def decision_tree_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn import tree

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    decision_tree_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', tree.DecisionTreeClassifier()),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [None],
        'clf__min_samples_split': [2, 3, 4],
        'clf__min_samples_leaf': [1, 2, 3, 4],
        'clf__min_weight_fraction_leaf': [0],
        'clf__max_features': [None, 'sqrt', 'log2'],
        'clf__max_leaf_nodes': [None, 2, 3, 4],
        'clf__min_impurity_decrease': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    }

    # Perform the grid search using all cores.
    decision_tree_classifier_clf = GridSearchCV(decision_tree_classifier_clf, parameters, cv=5, iid=False, n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    decision_tree_classifier_clf.fit(tweet_train, target_train)
    decision_tree_classifier_predictions = decision_tree_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(decision_tree_classifier_clf.cv_results_)
        log.debug("The shape of the Decision Tree Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Decision Tree Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Decision Tree Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, decision_tree_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Decision Tree Classifier Grid Search is: ")
    log.debug(np.mean(decision_tree_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def decision_tree_classifier():
    """
    Functions trains a Decision Tree Classifier.
    
    :return: none. 
    """
    from sklearn import tree

    decision_tree_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', tree.DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None,
                                            max_leaf_nodes=3, min_impurity_decrease=1e-5, min_samples_leaf=1,
                                            min_samples_split=2, min_weight_fraction_leaf=0)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        decision_tree_classifier_clf.fit(tweet_train, target_train)
        decision_tree_classifier_predictions = decision_tree_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(decision_tree_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Decision Tree Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("DecisionTree_classifier Metrics")
            log.debug(metrics.classification_report(target_test, decision_tree_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("DecisionTree_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, decision_tree_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Decision Tree Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Decision Tree Classifier:")
    make_predictions(decision_tree_classifier_clf)


################################################################################################################
def multi_layer_perceptron_classifier_grid_search():
    """
         Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

         :return: Nothing.
         """
    from sklearn.neural_network import MLPClassifier

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    mlp_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MLPClassifier(activation='logistic', alpha=1e-1, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=1e-1,
                              max_iter=1000, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='sgd', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1)],
        # 'tfidf__use_idf': (True, False),
        # 'clf__hidden_layer_sizes': [(15, 15), (50, 50)],
        'clf__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'clf__solver': ['lbfgs', 'sgd', 'adam'],
        'clf__alpha': [1e-1, 1e-2, 1e-4, 1e-6, 1e-8],
        # 'clf__batch_size': [5, 10, 20, 40, 80, 160],
        'clf__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'clf__learning_rate_init': [1e-1, 1e-3, 1e-5],
        # 'clf__power_t': [0.1, 0.25, 0.5, 0.75, 1.0],
        # 'clf__max_iter': [200, 400, 800, 1600],
        # 'clf_shuffle': [True, False],
        # 'clf__tol': [1e-1, 1e-2, 1e-4, 1e-6, 1e-8],
        # 'clf__momentum': [0.1, 0.3, 0.6, 0.9],
        # 'clf_nestesrovs_momentum': [True, False],
        # 'clf_early_stopping': [True, False],
        # 'clf__validation_fraction': [0.1, 0.2, 0.4],
        # 'clf_beta_1': [0.1, 0.2, 0.4, 0.6, 0.8],
        # 'clf_beta_2': [0.1, 0.2, 0.4, 0.6, 0.8],
        # 'clf_epsilon': [1e-1, 1e-2, 1e-4, 1e-8],
        # 'clf__n_iter_no_change': [1, 2, 4, 8, 16]

    }

    # Perform the grid search using all cores.
    mlp_classifier_clf = GridSearchCV(mlp_classifier_clf, parameters, cv=5, iid=False,
                                      n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    mlp_classifier_clf.fit(tweet_train, target_train)
    mlp_classifier_predictions = mlp_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(mlp_classifier_clf.cv_results_)
        log.debug("The shape of the Multi Layer Perceptron Neural Network Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Multi Layer Perceptron Neural Network Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Multi Layer Perceptron Neural Network Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, mlp_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Multi Layer Perceptron Neural Network Classifier Grid Search is: ")
    log.debug(np.mean(mlp_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def multi_layer_perceptron_classifier():
    """
    Function trains a Multi Layer Perceptron Neural Network Classifier.
    
    :return: none. 
    """
    from sklearn.neural_network import MLPClassifier

    mlp_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', MLPClassifier(activation='identity', alpha=1e-1, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=1e-1,
                              max_iter=1000, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.1,
                              validation_fraction=0.1, verbose=False, warm_start=False)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):
        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        # scaler.fit(tweet_train)
        # tweet_train_scaled = scaler.transform(tweet_train)
        # tweet_test_scaled = scaler.transform(tweet_test)

        mlp_classifier_clf.fit(tweet_train, target_train)
        mlp_classifier_predictions = mlp_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(mlp_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Decision Tree Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("MLP_classifier Metrics")
            log.debug(metrics.classification_report(target_test, mlp_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("MLP_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, mlp_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Multi Layer Perceptron Neural Network Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Multi Layer Perceptron Neural Network Classifier:")
    make_predictions(mlp_classifier_clf)


################################################################################################################
def logistic_regression_classifier_grid_search():
    """
       Function performs a exhaustive grid search to find the best hyper-parameters for use training the model.

       :return: Nothing.
       """
    from sklearn.linear_model import LogisticRegression

    # Create randomized training and test set using our dataset.
    create_training_and_test_set()

    logistic_regression_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=0, solver='lbfgs',
                                   multi_class='multinomial', n_jobs=-1)),
    ])

    from sklearn.model_selection import GridSearchCV

    # What parameters do we search for?
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__use_idf': (True, False),
        'clf__penalty': ['l2'],
        'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'clf__C': [0.2, 0.4, 0.6, 0.8, 1.0],
        'clf__fit_intercept': [True, False],
        'clf__class_weight': ['balanced', None],
        'clf__solver': ['saga', 'newton-cg', 'sag', 'lbfgs'],
        'clf__max_iter': [2000, 4000, 8000, 16000],
        'clf__multi_class': ['ovr', 'multinomial'],
    }

    # Perform the grid search using all cores.
    logistic_regression_classifier_clf = GridSearchCV(logistic_regression_classifier_clf, parameters, cv=5, iid=False,
                                                      n_jobs=-1)

    # Train and predict on optimal parameters found by Grid Search.
    logistic_regression_classifier_clf.fit(tweet_train, target_train)
    logistic_regression_classifier_predictions = logistic_regression_classifier_clf.predict(tweet_test)

    if debug_pipeline:
        # View all the information stored in the model after training it.
        classifier_results = pd.DataFrame(logistic_regression_classifier_clf.cv_results_)
        log.debug("The shape of the  Logistic Regression Classifier model's result data structure is:")
        log.debug(classifier_results.shape)
        log.debug(
            "The contents of the Logistic Regression Classifier model's result data structure is:")
        log.debug(classifier_results.head())

    # Display the optimal parameters.
    log.debug("The optimal parameters found for the Logistic Regression Classifier is:")
    for param_name in sorted(parameters.keys()):
        log.debug("%s: %r" % (param_name, logistic_regression_classifier_clf.best_params_[param_name]))
    log.debug("\n")

    # Display the accuracy we obtained using the optimal parameters.
    log.debug("Accuracy using Logistic Regression Classifier Grid Search is: ")
    log.debug(np.mean(logistic_regression_classifier_predictions == target_test))
    log.debug("\n")


################################################################################################################
def logistic_regression_classifier():
    """
    Function trains a Logistic Regression Classifier.
    
    :return: none. 
    """
    from sklearn.linear_model import LogisticRegression

    logistic_regression_classifier_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', LogisticRegression(C=1.0, class_weight=None, fit_intercept=False, max_iter=2000,
                                   multi_class='ovr', penalty='l2', solver='sag', tol=1e-1)),
    ])

    # Predict n iterations and calculate mean accuracy.
    mean_accuracy = 0.0
    iterations = 1000
    for index in range(0, iterations):

        # Create randomized training and test set using our dataset.
        create_training_and_test_set()

        logistic_regression_classifier_clf.fit(tweet_train, target_train)
        logistic_regression_classifier_predictions = logistic_regression_classifier_clf.predict(tweet_test)

        # Calculate the accuracy of our predictions.
        accuracy = np.mean(logistic_regression_classifier_predictions == target_test)

        if debug_classifier_iterations:
            # Measure accuracy.
            log.debug("\n")
            log.debug("Accuracy for test set predictions using Logistic Regression Classifier:")
            log.debug(str(accuracy))
            log.debug("\n")

            log.debug("LogisticRegression_classifier Metrics")
            log.debug(metrics.classification_report(target_test, logistic_regression_classifier_predictions,
                                                    target_names=['economic', 'environmental', 'social']))

            log.debug("LogisticRegression_classifier confusion matrix:")
            log.debug(metrics.confusion_matrix(target_test, logistic_regression_classifier_predictions))

        mean_accuracy += accuracy

    mean_accuracy = mean_accuracy / iterations
    log.debug("Logistic Regression Classifier:")
    log.debug("Mean accuracy over " + str(iterations) + " iterations is: " + str(mean_accuracy))
    log.debug("\n")

    # Make predictions using trained model.
    log.debug("Prediction statistics using Logistic Regression Classifier:")
    make_predictions(logistic_regression_classifier_clf)


################################################################################################################
def keras_deep_neural_network():
    """
    Function implements a Keras Deep Neural Network Model.
    TODO - most likely won't be implemented until summer research with much larger labeled datasets.
    :return: none. 
    """

    from keras.models import Sequential
    from keras import layers

    pass


################################################################################################################

############################################################################################
"""
Main function.  Execute the program.
"""

# Debug variable.
debug_main = 0

if __name__ == '__main__':

    import time

    start_time = time.time()

    # Call non-pipelined multinomial Naive Bayes classifier training function.
    # scikit_learn_multinomialnb_classifier_non_pipeline()

    # Call pipelined classifier training functions and grid search functions.
    # multinomial_naive_bayes_classifier_grid_search()
    multinomial_naive_bayes_classifier()
    # sgd_classifier_grid_search()
    sgd_classifier()
    # svm_support_vector_classification_grid_search()
    svm_support_vector_classification()
    # svm_linear_support_vector_classification_grid_search()
    svm_linear_support_vector_classification()
    # nearest_kneighbor_classifier_grid_search()
    nearest_kneighbor_classifier()
    # decision_tree_classifier_grid_search()
    decision_tree_classifier()
    # multi_layer_perceptron_classifier_grid_search()
    multi_layer_perceptron_classifier()
    # logistic_regression_classifier_grid_search()
    logistic_regression_classifier()

    end_time = time.time()

    if debug_pipeline:
        log.debug("The time taken to train the classifier(s) is:")
        total_time = end_time - start_time
        log.debug(str(total_time))
        log.debug("\n")

    # For debug purposes.
    # my_set = create_prediction_set()

############################################################################################
