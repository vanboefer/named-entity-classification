"""
This script processes raw data and generates features. The features are stored in pkl (pickle) and npz (numpy zipped archive) format in the data_processed directory.

The helper functions for this script are found in the utils package:
- extract_features
- embeddings_dict
"""

import timeit
import pickle
import scipy.sparse
import pandas as pd

import utils.extract_features as ef
import utils.embeddings_dict as ed

from config import ProjectPaths
from sklearn.feature_extraction import DictVectorizer


start = timeit.default_timer()

path_raw = ProjectPaths()['data_raw']
path_proc = ProjectPaths()['data_processed']
path_emb = ProjectPaths()['word_embeddings']

#######################################################
# SELECT TRAIN AND TEST DATA (user input)
#######################################################
# define the input data files (train and test)
files = path_raw.glob('*.*')
dct_files = dict()
print(f'Files in data_raw directory:')
for idx, filepath in enumerate(files):
    print(f'[{idx}] - {filepath.stem}')
    dct_files[idx] = filepath
print()
print(f'Select indexes for training and test files (0 to {len(dct_files) - 1}):')
train_idx = int(input('Index of training data file: '))
test_idx = int(input('Index of test data file: '))

#######################################################
# PROCESS RAW DATA TO FEATURES
#######################################################
# create df's with features
train_df = ef.data2features(dct_files[train_idx])
test_df = ef.data2features(dct_files[test_idx])

print('DataFrames are loaded!')

#######################################################
# ADD PRE-TRAINED WORD EMBEDDINGS
#######################################################
# create a pkl file with a dictionary mapping each lemma in train and test data
# to its embedding vector (from a file of pre-trained embeddings)
embeddings_pkl = path_proc / 'data_embeddings.pkl'
words_no_embed = path_proc / 'words_without_embeddings.txt'
if not embeddings_pkl.exists():
    print('Creating embeddings pkl...')
    ed.embed_pkl(path_emb,
                 train_df,
                 test_df,
                 embeddings_pkl,
                 words_no_embed)
    print('Embeddings pkl is created!')

# load fastText pre-trained word embeddings of the lemmas found in data (train + test)
with open(embeddings_pkl, 'rb') as infile:
    data_embeddings = pickle.load(infile)
print('Embeddings pkl is loaded!')

# create df with the embeddings (lemmas as index)
data_embeddings_df = pd.DataFrame.from_dict(data_embeddings, orient='index')
data_embeddings_df.index.rename('lemma', inplace=True)

# add the embeddings info to train_df and test_df;
# embeddings for current lemma (cl), previous lemma (pl) and next lemma (nl) are added to each row (900 columns in total)
merg_dict = {'lemma': 'cl', 'prev_lemma': 'pl', 'next_lemma': 'nl'}
for key in merg_dict:
    lemmatype = merg_dict[key]
    cols = [lemmatype + str(idx) for idx, col in enumerate(data_embeddings_df.columns)]
    data_embeddings_df.columns = cols
    train_df = pd.merge(train_df,
                        data_embeddings_df,
                        how='left',
                        left_on=key, right_index=True)
    test_df = pd.merge(test_df,
                       data_embeddings_df,
                       how='left',
                       left_on=key, right_index=True)

print('Embeddings are added to the train and test df\'s!')

#######################################################
# PREPARE FEATURES FOR SVM, NB, LogReg CLASSIFIERS
#######################################################
# feat0 = lemma (str), prev_lemma (str), next_lemma (str)
feat0_cols = ['lemma', 'prev_lemma', 'next_lemma']

# feat1 = lemma (embed), prev_lemma (embed), next_lemma (embed)
feat1_cols = [col for col in train_df.columns if col[:2] in merg_dict.values()]

# feat2 = lemma (str), prev_lemma (str), next_lemma (str), pos, prev_pos, next_pos, word shape
feat2_cols = ['lemma', 'prev_lemma', 'next_lemma', 'pos', 'prev_pos', 'next_pos', 'shape']

# feat2 = lemma (embed), prev_lemma (embed), next_lemma (embed), pos, prev_pos, next_pos, word_shape
feat3_cols = [col for col in feat2_cols + feat1_cols if col not in feat0_cols]

feat_select = {
            'feat0':feat0_cols,
            'feat1':feat1_cols,
            'feat2':feat2_cols,
            'feat3':feat3_cols,
            }

# prepare the four sets of features and write them to npz files
print('Preparing X_train and X_test...')
for item in feat_select:
    vec = DictVectorizer()

    # train
    X_train_list = train_df[feat_select[item]].to_dict(orient='records')
    X_train = vec.fit_transform(X_train_list)
    scipy.sparse.save_npz(path_proc / (item + '_X_train.npz'), X_train)

    # test
    X_test_list = test_df[feat_select[item]].to_dict(orient='records')
    X_test = vec.transform(X_test_list)
    scipy.sparse.save_npz(path_proc / (item + '_X_test.npz'), X_test)

    print(f'{item} is ready!')

# write the gold NER tags to pkl files
y_train = list(train_df.NER_gold)
y_test = list(test_df.NER_gold)

files_to_save = {
                'y_train.pkl': y_train,
                'y_test.pkl': y_test,
                }

for file in files_to_save:
    with open(path_proc / file, 'wb') as f:
        pickle.dump(files_to_save[file], f)

print('y_train and y_test are ready!')

#######################################################
# PREPARE FEATURES FOR HMM
#######################################################
# prepare hmm features (sentences, tag_set, lemma_set)
hmm_train = ef.hmm_feat(train_df)
hmm_test = ef.hmm_feat(test_df)

files_to_save = {
                'hmm_train.pkl': hmm_train,
                'hmm_test.pkl': hmm_test,
                }

for file in files_to_save:
    with open(path_proc / file, 'wb') as f:
        pickle.dump(files_to_save[file], f)

print('HMM features are ready!')

#######################################################
# PREPARE FEATURES FOR CRF
#######################################################
# prepare crf features (sents, tags_sents)

# feat0 setting:
crf_feat0 = ['lemma', 'prev_lemma', 'next_lemma', 'NER_gold']
crf_train_feat0 = ef.crf_feat(train_df[crf_feat0])
crf_test_feat0 = ef.crf_feat(test_df[crf_feat0])

# feat2 setting:
crf_feat2 = ['lemma', 'prev_lemma', 'next_lemma', 'pos', 'prev_pos', 'next_pos', 'shape', 'NER_gold']
crf_train_feat2 = ef.crf_feat(train_df[crf_feat2])
crf_test_feat2 = ef.crf_feat(test_df[crf_feat2])

files_to_save = {
                'crf_train_feat0.pkl': crf_train_feat0,
                'crf_test_feat0.pkl': crf_test_feat0,
                'crf_train_feat2.pkl': crf_train_feat2,
                'crf_test_feat2.pkl': crf_test_feat2,
                }

for file in files_to_save:
    with open(path_proc / file, 'wb') as f:
        pickle.dump(files_to_save[file], f)

print('CRF features are ready!')

stop = timeit.default_timer()
ex_time = stop - start
print(f'Script executed in {ex_time/60:.2f} minutes.')
