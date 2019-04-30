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

TODO - resolve SettingWithCopyWarning:

###########################################################
Resources Used:

Refer to original un-cleaned version.

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
from SLO_TBL_Tweet_Preprocessor_Specialized import tweet_dataset_preprocessor_1, tweet_dataset_preprocessor_3

#############################################################

log.basicConfig(level=log.DEBUG)
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
debug = True

################################################################################################################
################################################################################################################

# Call Tweet pre-processing module.
tweet_dataframe_processed = tweet_dataset_preprocessor_1()

# Assign column names.
tweet_dataframe_processed_column_names = ['Tweet', 'SLO']

# Create input features.
selected_features = tweet_dataframe_processed[tweet_dataframe_processed_column_names]
processed_features = selected_features.copy()

if debug:
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

from sklearn.model_selection import train_test_split

# Split feature and target set into training and test sets for each set.
tweet_train, tweet_test, target_train, target_test = train_test_split(slo_feature_set, slo_target_set, test_size=0.33,
                                                                      random_state=42)
if debug:
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
target_train_DEcoded = target_label_encoder.inverse_transform(target_train_encoded)
target_test_DEcoded = target_label_encoder.inverse_transform(target_test_encoded)

if debug:
    log.debug("Encoded target training labels:")
    log.debug(target_train_encoded)
    log.debug("Decoded target training labels:")
    log.debug(target_train_DEcoded)
    log.debug("\n")
    log.debug("Encoded target test labels:")
    log.debug(target_test_encoded)
    log.debug("Decoded target test labels:")
    log.debug(target_test_DEcoded)
    log.debug("\n")


#######################################################
def scikit_learn_multinomialnb_classifier_non_pipeline():
    """
    Function trains a Multinomial Naive Bayes Classifier without using a Pipeline.

    Note: Implemented for educational purposes - so I can see the manual workflow.

    :return: none.
    """
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

    ################################################################################################################

    from sklearn.naive_bayes import MultinomialNB

    # Train the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb = MultinomialNB().fit(tweet_train_encoded_tfidf, target_train_encoded)
    # Predict using the Multinomial Naive Bayes Classifier.
    clf_multinomial_nb_predict = clf_multinomial_nb.predict(tweet_test_encoded_tfidf)

    from sklearn.metrics import accuracy_score

    print("MultinomialNB Classifier accuracy using accuracy_score() function : ",
          accuracy_score(target_test_encoded, clf_multinomial_nb_predict, normalize=True))
    print()

    # Another method of obtaining accuracy metric.
    print("Accuracy for test set predictions using multinomialNB:")
    print(str(np.mean(clf_multinomial_nb_predict == target_test_encoded)))
    print()

    # View the results as Tweet => predicted topic classification label.
    for doc, category in zip(tweet_test, clf_multinomial_nb_predict):
        log.debug('%r => %s' % (doc, category))

    ################################################################################################################
    """
    Make predictions using pre-processed and post-processed Tweets from CMU Tweet Tagger.
    
    Note: Probably won't be the best generalization to new data as the vocabulary between these two different 
    datasets could be drastically different.
    
    """

    # Call Tweet pre-processing module.
    processed_features_cmu = tweet_dataset_preprocessor_3()

    #######################################################

    # Vectorize the categorical data for use in predictions.
    tweet_predict_encoded = vectorizer.transform(processed_features_cmu)

    if debug:
        log.debug("Vectorized tweet predictions set:")
        log.debug(tweet_predict_encoded)
        log.debug("\n")
        log.debug("Shape of the tweet predictions set:")
        log.debug(tweet_predict_encoded.shape)
        log.debug("\n")

    # Convert to term frequencies.
    tweet_predict_encoded_tfidf = tfidf_transformer.transform(tweet_predict_encoded)

    if debug:
        log.debug("Vectorized tweet predictions set term frequencies down-sampled:")
        log.debug(tweet_predict_encoded_tfidf)
        log.debug("\n")
        log.debug("Shape of the tweet predictions set term frequencies: ")
        log.debug(tweet_predict_encoded_tfidf.shape)
        log.debug("\n")

    # Generalize to new data and predict.
    tweet_generalize_new_data_predictions = clf_multinomial_nb.predict(tweet_predict_encoded_tfidf)

    prediction_df = pd.DataFrame(tweet_generalize_new_data_predictions)
    print("The shape of our prediction dataframe:")
    print(prediction_df.shape)
    print("The columns of our prediction dataframe:")
    print(prediction_df.head())
    print("Samples from our prediction dataframe:")
    print(prediction_df.tail())

    # View the results as Tweet => predicted topic classification label.
    # Note: There are 500k+ Tweets in this dataset, don't log.debug out unless you want a very long output list.
    # for doc, category in zip(processed_features_cmu, tweet_generalize_new_data_predictions):
    #     log.debug('%r => %s' % (doc, category))

    ################################################################################################################

    #######################################################

    def grid_search():
        """
        Helper function defines a grid search for optimal hyper parameters.
        TODO - implement grid search.
        :return: optimzal hyper parameters.
        """

        from sklearn.model_selection import GridSearchCV

        # What parameters do we search for?
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }

        # Perform the grid search using all cores.
        gs_clf = GridSearchCV(clf_multinomial_nb, parameters, cv=5, iid=False, n_jobs=-1)

        gs_clf_fit = gs_clf.fit(tweet_train, target_train)
        gs_clf_predict = gs_clf_fit.predict(tweet_test)

        pass


################################################################################################################
def multinomial_naive_bayes_classifier():
    """
    Functions trains a Multinomial Naive Bayes Classifier.

    :return: none.
    """
    from sklearn.naive_bayes import MultinomialNB

    multinomial_nb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('multinomialNB', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)),
    ])

    multinomial_nb_clf.fit(tweet_train, target_train)
    multinomial_nb_predictions = multinomial_nb_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using Multinomial Naive Bayes Classifier:")
    print(str(np.mean(multinomial_nb_predictions == target_test)))
    print()

    print("Multinomial Naive Bayes Classifier Metrics")
    print(metrics.classification_report(target_test, multinomial_nb_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("Multinomial Naive Bayes Classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, multinomial_nb_predictions))


################################################################################################################
def sgd_classifier():
    """
    Function trains a Stochastic Gradient Descent Classifier.
    
    :return: none.
    """
    from sklearn.linear_model import SGDClassifier

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

    sgd_classifier_clf.fit(tweet_train, target_train)
    sgd_classifier_predictions = sgd_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using SGD_classifier:")
    print(str(np.mean(sgd_classifier_predictions == target_test)))
    print()

    print("SGD_classifier Classifier Metrics")
    print(metrics.classification_report(target_test, sgd_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("SGD_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, sgd_classifier_predictions))


################################################################################################################
def svm_support_vector_classification():
    """
    Functions trains a Support Vector Machine - Support Vector Classification Classifier.
    
    :return: none.
    """
    from sklearn import svm

    svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)),
    ])

    svc_classifier_clf.fit(tweet_train, target_train)
    svc_classifier_predictions = svc_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using SVC_classifier:")
    print(str(np.mean(svc_classifier_predictions == target_test)))
    print()

    print("SVC_classifier Metrics")
    print(metrics.classification_report(target_test, svc_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("SVC_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, svc_classifier_predictions))


################################################################################################################
def svm_linear_support_vector_classification():
    """"
    Function trains a Support Vector Machine - Linear Support Vector Classification Classifier.
    
    :return: none.
    """
    from sklearn import svm

    linear_svc_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                              verbose=0)),
    ])

    linear_svc_classifier_clf.fit(tweet_train, target_train)
    linear_svc_classifier_predictions = linear_svc_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using LinearSVC_classifier:")
    print(str(np.mean(linear_svc_classifier_predictions == target_test)))
    print()

    print("LinearSVC_classifier Metrics")
    print(metrics.classification_report(target_test, linear_svc_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("LinearSVC_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, linear_svc_classifier_predictions))


################################################################################################################
def nearest_kneighbor_classifier():
    """
    Function trains a Nearest Neighbor - KNeighbor Classifier.
    
    :return: none. 
    """
    from sklearn.neighbors import KNeighborsClassifier

    k_neighbor_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=3)),
    ])

    k_neighbor_classifier_clf.fit(tweet_train, target_train)
    k_neighbor_classifier_predictions = k_neighbor_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using KNeighbor_classifier:")
    print(str(np.mean(k_neighbor_classifier_predictions == target_test)))
    print()

    print("KNeighbor_classifier Metrics")
    print(metrics.classification_report(target_test, k_neighbor_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("KNeighbor_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, k_neighbor_classifier_predictions))


################################################################################################################
def decision_tree_classifier():
    """
    Functions trains a Decision Tree Classifier.
    
    :return: none. 
    """
    from sklearn import tree

    decision_tree_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', tree.DecisionTreeClassifier(random_state=0)),
    ])

    decision_tree_classifier_clf.fit(tweet_train, target_train)
    decision_tree_classifier_predictions = decision_tree_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using DecisionTree_classifier:")
    print(str(np.mean(decision_tree_classifier_predictions == target_test)))
    print()

    print("DecisionTree_classifier Metrics")
    print(metrics.classification_report(target_test, decision_tree_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("DecisionTree_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, decision_tree_classifier_predictions))


################################################################################################################
def multi_layer_perceptron_classifier():
    """
    Function trains a Multi Layer Perceptron Neural Network Classifier.
    
    :return: none. 
    """
    from sklearn.neural_network import MLPClassifier

    mlp_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MLPClassifier(activation='relu', alpha=1e-5, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=True,
                              epsilon=1e-08, hidden_layer_sizes=(15, 15, 15),
                              learning_rate='constant', learning_rate_init=0.001,
                              max_iter=1000, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)),
    ])

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # scaler.fit(tweet_train)
    # tweet_train_scaled = scaler.transform(tweet_train)
    # tweet_test_scaled = scaler.transform(tweet_test)

    mlp_classifier_clf.fit(tweet_train, target_train)
    mlp_classifier_predictions = mlp_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using MLP_classifier:")
    print(str(np.mean(mlp_classifier_predictions == target_test)))
    print()

    print("MLP_classifier Metrics")
    print(metrics.classification_report(target_test, mlp_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("MLP_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, mlp_classifier_predictions))


################################################################################################################
def logistic_regression_classifier():
    """
    Function trains a Logistic Regression Classifiers.
    
    :return: none. 
    """
    from sklearn.linear_model import LogisticRegression

    logistic_regression_classifier_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(random_state=0, solver='lbfgs',
                                   multi_class='multinomial')),
    ])

    logistic_regression_classifier_clf.fit(tweet_train, target_train)
    logistic_regression_classifier_predictions = logistic_regression_classifier_clf.predict(tweet_test)

    # Measure accuracy.
    print()
    print("Accuracy for test set predictions using LogisticRegression_classifier:")
    print(str(np.mean(logistic_regression_classifier_predictions == target_test)))
    print()

    print("LogisticRegression_classifier Metrics")
    print(metrics.classification_report(target_test, logistic_regression_classifier_predictions,
                                        target_names=['economic', 'environmental', 'social']))

    print("LogisticRegression_classifier confusion matrix:")
    print(metrics.confusion_matrix(target_test, logistic_regression_classifier_predictions))


################################################################################################################
def keras_deep_neural_network():
    """
    Function implements a Keras Deep Neural Network Model.
    
    :return: none. 
    """

    from keras.models import Sequential
    from keras import layers


################################################################################################################

############################################################################################
"""
Main function.  Execute the program.
"""

# Debug variable.
debug_main = 0

if __name__ == '__main__':
    # Call non-pipelined multinomial Naive Bayes classifier training function.
    scikit_learn_multinomialnb_classifier_non_pipeline()

    # Call pipelined classifier training functions.
    multinomial_naive_bayes_classifier()
    sgd_classifier()
    svm_support_vector_classification()
    svm_linear_support_vector_classification()
    nearest_kneighbor_classifier()
    decision_tree_classifier()
    multi_layer_perceptron_classifier()
    logistic_regression_classifier()

############################################################################################
