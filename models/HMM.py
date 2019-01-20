"""
Hidden Markov Model Tagger
==========================
The code is based on: http://www.nltk.org/_modules/nltk/tag/hmm.html
"""

import pickle
import itertools

from nltk.tag import hmm
from nltk.tag.hmm import HiddenMarkovModelTagger
from nltk.probability import LidstoneProbDist


def supervised_hmm(train_pkl, test_pkl):
    """
    Run a supervised training with NLTK HMM tagger.
    Test the tagger on test data and return y_pred.

    :param train_pkl: pkl file with training data in the required format
    :param test_pkl: pkl file with test data in the required format
    :return: y_pred (list)
    """

    # load from files
    with open(train_pkl, 'rb') as infile:
        train_sents, train_tags, train_lemmas = pickle.load(infile)

    with open(test_pkl, 'rb') as infile:
        test_sents, test_tags, test_lemmas = pickle.load(infile)

    # training
    trainer = hmm.HiddenMarkovModelTrainer(train_tags, train_lemmas)

    sup_hmm = trainer.train_supervised(train_sents,
                                   estimator=lambda fd,
                                   bins: LidstoneProbDist(fd, 0.1, bins),)

    # testing
    y_pred = custom_test(sup_hmm, test_sents)

    return y_pred


def custom_test(hmm_tagger, test_sents, **kwargs):
    """
    Test the HiddenMarkovModelTagger instance.
    This customized version returns y_pred.

    :param hmm_tagger: instance of the HiddenMarkovModelTagger class
    :param test_sents: a list of lists, where each list is a sentence; each token in the sentence is represented by a tuple (lemma, tag)
    :return: y_pred (list)
    """

    def lemmas(sent):
        return [lemma for (lemma, tag) in sent]

    def tags(sent):
        return [tag for (lemma, tag) in sent]

    def flatten(seq):
        return list(itertools.chain(*seq))

    test_sents = hmm_tagger._transform(test_sents)
    predicted_sequence = list(map(hmm_tagger._tag, map(lemmas, test_sents)))

    y_pred = flatten(map(tags, predicted_sequence))

    return y_pred
