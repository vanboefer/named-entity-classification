"""
Logistic Regression
===================
"""

import pickle
import scipy.stats
import scipy.sparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def log_reg(X_train_npz, X_test_npz, y_train_pkl, optimize=False):
    """
    Run a Logistic Regression classifier, incl. optional optimization of C.

    :param X_train_npz: npz file with sparse csr matrix
    :param y_train_pkl: pkl file with a list of gold tags
    :param X_test_npz: npz file with sparse csr matrix
    :param optimize: if True, perform optimization of C (default=False)
    :return: best parameter values (dict), y_pred (list)
    """

    # load from files
    X_train = scipy.sparse.load_npz(X_train_npz)

    with open(y_train_pkl, 'rb') as f:
        y_train = pickle.load(f)

    X_test = scipy.sparse.load_npz(X_test_npz)

    # training
    if optimize == True:
        # parameter optimization: find the best value for C
        lr = LogisticRegression(solver='liblinear', multi_class='auto')

        params_space = {'C': scipy.stats.expon(scale=10)}

        # search
        rs = RandomizedSearchCV(lr, params_space,
                                cv=3,
                                n_jobs=-1,
                                scoring='f1_weighted')

        rs.fit(X_train, y_train)

        # best params
        best_params = rs.best_params_

        lr = rs.best_estimator_

    else:
        lr = LogisticRegression(solver='liblinear', multi_class='auto')
        lr.fit(X_train, y_train)

        best_params = {'C': 1.0}

    # testing
    y_pred = lr.predict(X_test)

    return best_params, y_pred
