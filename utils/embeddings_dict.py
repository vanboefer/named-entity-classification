"""
This module contains helper functions for creating the data_embeddings.pkl file. This file contains a dictionary where each lemma in the train and test data is mapped to a corresponding pre-trained fastText word embedding (https://fasttext.cc/docs/en/english-vectors.html).

Lemmas that do not have an embedding in this pre-trained database, will have embeddings of all-zeros in the dictionary.

The module is used in the create_X_y.py script.
"""

import io
import pickle
import pandas as pd
import numpy as np


def embed_pkl(vec_file, train_df, test_df, pkl_file, txt_file):
    """
    Create a dictionary mapping each lemma in train and test data to its embedding vector (from a file of pre-trained embeddings).
    Lemmas that are not found in the vec_file will have an all-zeros vector.
    Write the dictionary to a pkl file; write the list of words for which there are no embeddings to a txt file.
    :param vec_file: path to .vec file
    :param train_df: DataFrame with a 'lemma' column containing all lemmas of the train data
    :param test_df: DataFrame with a 'lemma' column containing all lemmas of the test data
    :param pkl_file: path to the output pkl file with the dictionary
    :param txt_file: path to the output txt file with words for which there are no embeddings
    :return: None (two output files are created in the indicated directory)
    """

    embed_dict = load_vectors(vec_file)

    # create a set of all lemmas in train and test data --> cast it into a list (so that there is order)
    # BOS and EOS are added because they appear in prev_lemma and next_lemma features
    words_set = set(train_df.lemma)
    words_set.update(set(test_df.lemma))
    words_set.add('BOS')
    words_set.add('EOS')
    words = list(words_set)

    # create embedding matrix for all lemmas in train and test data; words not found in embed_dict will be all-zeros
    # (based on: https://www.kaggle.com/vsmolyakov/keras-cnn-with-fasttext-embeddings)
    embed_dim = 300
    no_words = len(words)
    words_not_found = []

    embedding_matrix = np.zeros((no_words, embed_dim))

    for i, word in enumerate(words):
        if i >= no_words:
            continue
        embedding_vector = embed_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = list(embedding_vector)
        else:
            words_not_found.append(word)

    # create a dictionary mapping each lemma to its embedding vector
    data_embeddings = dict()
    for i, word in enumerate(words):
        data_embeddings[word] = embedding_matrix[i]

    # write the dictionary to a pkl file
    with open(pkl_file, 'wb') as outfile:
        pickle.dump(data_embeddings, outfile)

    # write the list of words for which there are no embeddings to a txt file
    with open(txt_file, 'w') as outfile:
        outfile.write('\n'.join(words_not_found))


def load_vectors(fname):
    """
    based on: https://fasttext.cc/docs/en/english-vectors.html

    Read a file with pre-trained word embeddings and return a dictionary: {token:map object}.
    The map object can be passed to e.g. list() to return the vector.
    :param fname: path to .vec file
    :return: dict
    """

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embed_dict = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embed_dict[tokens[0]] = map(float, tokens[1:])
    return embed_dict
