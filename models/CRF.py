"""
Conditional Random Field
========================
The code is based on: https://github.com/TeamHG-Memex/sklearn-crfsuite/blob/master/docs/CoNLL2002.ipynb
"""

import pickle
import scipy.stats

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV


def crf(train_pkl, test_pkl, optimize=False):
    """
    Run a CRF classifier, incl. optional optimization of c1 and c2.

    :param train_pkl: pkl file with training data in the required format
    :param test_pkl: pkl file with test data in the required format
    :return:
        - best parameter values (dict)
        - y_pred (list of lists)
        - y_pred_converted (list)
    """

    # load from files
    with open(train_pkl, 'rb') as infile:
        X_train, y_train = pickle.load(infile)

    with open(test_pkl, 'rb') as infile:
        X_test, y_test = pickle.load(infile)

    # training
    if optimize == True:
        # parameter optimization: find the best values for c1 and c2
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )

        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }

        # the metric for evaluation
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

        # search
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                n_jobs=-1,
                                scoring=f1_scorer)

        rs.fit(X_train, y_train)

        # best params
        best_params = rs.best_params_

        crf = rs.best_estimator_

    else:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X_train, y_train)

        best_params = {'c1': 0.1, 'c2': 0.1}

    # testing
    y_pred_crf = crf.predict(X_test)

    # convert y_pred to one big list (remove sentence groupings)
    y_pred = list()
    for lst in y_pred_crf:
        for item in lst:
            y_pred.append(item)

    return best_params, y_pred_crf, y_pred
