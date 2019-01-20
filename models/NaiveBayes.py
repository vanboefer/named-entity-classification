"""
Naive Bayes
===========
"""

import pickle
import scipy.stats
import scipy.sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import RandomizedSearchCV


def mnb(X_train_npz, X_test_npz, y_train_pkl, optimize=False):
    """
    Run a Multinomial Naive Bayes classifier, incl. optional optimization of alpha.

    :param X_train_npz: npz file with sparse csr matrix
    :param y_train_pkl: pkl file with a list of gold tags
    :param X_test_npz: npz file with sparse csr matrix
    :param optimize: if True, perform optimization of alpha (default=False)
    :return: best parameter values (dict), y_pred (list)
    """
    # load from files
    X_train = scipy.sparse.load_npz(X_train_npz)

    with open(y_train_pkl, 'rb') as f:
        y_train = pickle.load(f)

    X_test = scipy.sparse.load_npz(X_test_npz)

    # training
    if optimize == True:
        # parameter optimization: find the best value for alpha
        mnb = MultinomialNB()

        params_space = {'alpha': scipy.stats.expon(scale=1)}

        # search
        rs = RandomizedSearchCV(mnb, params_space,
                                cv=3,
                                n_jobs=-1,
                                scoring='f1_weighted')

        rs.fit(X_train, y_train)

        # best params
        best_params = rs.best_params_

        mnb = rs.best_estimator_

    else:
        mnb = MultinomialNB()
        mnb.fit(X_train, y_train)

        best_params = {'alpha': 1.0}

    # testing
    y_pred = mnb.predict(X_test)

    return best_params, y_pred


def cnb(X_train_npz, X_test_npz, y_train_pkl, optimize=False):
    """
    Run a Complement Naive Bayes classifier, incl. optional optimization of alpha.

    :param X_train_npz: npz file with sparse csr matrix
    :param y_train_pkl: pkl file with a list of gold tags
    :param X_test_npz: npz file with sparse csr matrix
    :param optimize: if True, perform optimization of alpha (default=False)
    :return: best parameter values (dict), y_pred (list)
    """

    # load from files
    X_train = scipy.sparse.load_npz(X_train_npz)

    with open(y_train_pkl, 'rb') as f:
        y_train = pickle.load(f)

    X_test = scipy.sparse.load_npz(X_test_npz)

    # training
    if optimize == True:
        # parameter optimization: find the best value for alpha
        cnb = ComplementNB()

        params_space = {'alpha': scipy.stats.expon(scale=1)}

        # search
        rs = RandomizedSearchCV(cnb, params_space,
                                cv=3,
                                n_jobs=-1,
                                scoring='f1_weighted')

        rs.fit(X_train, y_train)

        # best params
        best_params = rs.best_params_

        cnb = rs.best_estimator_

    else:
        cnb = ComplementNB()
        cnb.fit(X_train, y_train)

        best_params = {'alpha': 1.0}

    # testing
    y_pred = cnb.predict(X_test)

    return best_params, y_pred
