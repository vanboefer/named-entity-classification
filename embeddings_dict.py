import io
import pickle
import pandas as pd
import numpy as np

def load_vectors(fname):
    """
    based on: https://fasttext.cc/docs/en/english-vectors.html

    Read a file with pre-trained word embeddings and return a dictionary: {token:map object}.
    The map object can be passed to e.g. list() to return the vector.
    :param fname: path to .vec file
    :returns: dict
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    embed_dict = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        embed_dict[tokens[0]] = map(float, tokens[1:])
    return embed_dict

# load pre-trained fastText vectors (downloaded from: https://fasttext.cc/docs/en/english-vectors.html)
embed_dict = load_vectors(r'C:\Users\User\OneDrive\fastText_embeddings_EN\wiki-news-300d-1M.vec\wiki-news-300d-1M.vec')

# load data features df from csv
train_df = pd.read_csv('train_features.tsv', sep='\t', keep_default_na=False)
test_df = pd.read_csv('test_features.tsv', sep='\t', keep_default_na=False)

# create a set of all lemmas in train and test data --> cast it into a list (so that there is order)
words_set = set(train_df.lemma)
words_set.update(set(test_df.lemma))
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
with open('data_embeddings.pkl', 'wb') as outfile:
    pickle.dump(data_embeddings, outfile)
