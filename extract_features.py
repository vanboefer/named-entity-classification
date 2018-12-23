import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer


def data2features(infilepath, outfilepath, token_column=0, ner_column=3, sep=' ', sent_seq=False):
    """
    Process a data file with tokens and gold NER tags; output a csv file with the following features:
        - token
        - gold NER label
        - POS-tag
        - lemma
        - word shape (lowcase / upcase_BOS (beginning of sentence) / upcase_IN (middle of sentence) / all_caps / other)
        - previous lemma (t-1); BOS if current token is the beginning of a sentecne
        - next lemma (t+1); EOS if current token is the end of a sentecne
        - previous POS-tag (t-1); BOS if current token is the beginning of a sentecne
        - next POS-tag (t+1); EOS if current token is the end of a sentecne

    :param infilepath: path to a data file where
        - each token is in a separate row,
        - sentences (incl. titles) are separated by an empty row,
        - there are at least two columns: token and gold NER tag
    :param outfilepath: path to the output csv file with the features
    :param token_column: integer indicating the index of the column with the tokens (default=0)
    :param ner_column: integer indicating the index of the column with the NER tags (default=3)
    :sep: the separator used in the infile (default=' ')
    :returns: None (a csv file with the features is created in the indicated directory)
    """

    # load data into a DataFrame
    # a temporary placeholder | indicates sentence boundaries; it will be removed at a later step
    data_df = load_data(infilepath, token_column=token_column, ner_column=ner_column, sep=sep)

    # clean NER tags (remove I- and B- distinction)
    data_df['NER_gold'] = data_df['NER_gold'].apply(lambda x: clean_ner_tags(x))

    # add POS-tags (NLTK tagger)
    data_df['pos'] = pos_tagger(data_df['token'])

    # add lemmas (NLTK WordNet Lemmatizer + lower case everything)
    data_df['lemma'] = data_df['token'].apply(lambda x: lemmatizer(x))

    # add word shapes
    data_df['shape'] = word_shape(data_df)

    # add previos / next lemma
    data_df['prev_lemma'] = data_df['lemma'].shift(1).fillna('|')
    data_df['next_lemma'] = data_df['lemma'].shift(-1).fillna('|')

    # add previous / next POS-tag
    data_df['prev_pos'] = data_df['pos'].shift(1).fillna('|')
    data_df['next_pos'] = data_df['pos'].shift(-1).fillna('|')

    # create BOS/EOS tags where needed
    data_df['prev_lemma'] = data_df['prev_lemma'].apply(lambda x: 'BOS' if x == '|' else x)
    data_df['prev_pos'] = data_df['prev_pos'].apply(lambda x: 'BOS' if x == '|' else x)
    data_df['next_lemma'] = data_df['next_lemma'].apply(lambda x: 'EOS' if x == '|' else x)
    data_df['next_pos'] = data_df['next_pos'].apply(lambda x: 'EOS' if x == '|' else x)

    # remove rows with " as the token
    data_df = data_df.drop(data_df.query("token == '\"'").index, axis=0)

    if sent_seq == True:
        pass
    else:
        # remove rows with the placeholder |
        data_df = data_df.drop(data_df.query('token == "|"').index, axis=0)

        # write the csv outfile
        df2csv(data_df, outfilepath)

    return None


def load_data(filepath, token_column=0, ner_column=3, sep=' '):
    """
    Process a data file with tokens and gold NER tags; output a pandas DataFrame.
    :param filepath: path to a data file where
        - each token is in a separate row,
        - sentences (incl. titles) are separated by an empty row,
        - there are at least two columns - token and gold NER tag
    :param token_column: integer indicating the index of the column with the tokens (default=0)
    :param ner_column: integer indicating the index of the column with the NER tags (default=3)
    :sep: the separator used in the data file (default=' ')
    :returns: DataFrame with the columns ['token', 'NER_gold']
    """
    with open(filepath, 'r') as f:
        text = f.read()

    data_lst = list()
    for item in text.split('\n'):
        if item == '-DOCSTART- -X- O O':
            continue
        elif item == '':
            token = '|'     # this is a temporary placeholder that indicates sentence boundaries
            ner_tag = '-'
        else:
            token = item.split(sep=sep)[token_column]
            ner_tag = item.split(sep=sep)[ner_column]
        data_lst.append([token, ner_tag])

    data_df = pd.DataFrame(data_lst, columns=['token', 'NER_gold'])
    return data_df


def clean_ner_tags(ner_tag):
    """
    Check if a NER label starts with 'I-' or 'B-'; if so, remove 'I-' and 'B-'.
    :param ner_tag: a string
    :returns: a string without 'I-' and 'B-' in the beginning
    """
    if ner_tag[:2] == 'I-' or ner_tag[:2] == 'B-':
        ner_tag = ner_tag[2:]
    return ner_tag


def pos_tagger(tokens):
    """
    Process a list/series of tokens with NLTK pos-tagger; output a list of pos-tags.
    :param tokens: a container with tokens (list or pandas series)
    :return pos_list: list of pos-tags corresponding to the input tokens
    """
    pos_list = list()
    for item in nltk.pos_tag(tokens):
        if item[0] == '|':
            pos_list.append('|')
        else:
            pos_list.append(item[1])
    return pos_list


def lemmatizer(token):
    """
    Process a token with NLTK WordNet Lemmatizer; output the lemma of the token in lower case.
    :param token: a string
    :returns: the lemma corresponding to the token, in lower case
    """
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(token).lower()
    return lemma


def word_shape(df):
    """
    Process a sequence of tokens (first column in the input DataFrame); assign a word shape to each token:
        - lowcase
        - upcase_BOS (token is in the beginning of a sentence and starts with a capital letter)
        - upcase_IN (token is in the middle of a sentence and starts with a capital letter)
        - all_caps
        - other
    :param df: DataFrame where the first column (index=0) is a token
    :returns: a list of word shapes
    """
    shape_list = list()
    for i in range(0, len(df)):
        token = df.iloc[i, 0]
        if i == 0:
            prev_token = '|'
        else:
            prev_token = df.iloc[i-1, 0]

        if token.islower():
            shape = 'lowcase'
        elif token.istitle():
            if prev_token == '|':
                shape = 'upcase_BOS'
            else:
                shape = 'upcase_IN'
        elif token.isupper():
            shape = 'all_caps'
        else:
            shape = 'other'
        shape_list.append(shape)

    return shape_list


def df2csv(df, outfilepath):
    """
    Write a DataFrame to a csv file (sep='\t'), without an index column.
    :param df: DataFrame
    :param outfilepath: path to the output file
    :returns: None (a csv file is created in the indicated directory)
    """
    df.to_csv(path_or_buf=outfilepath, sep='\t', index=False)
    return None


if __name__ == '__main__':
    data2features('data/train_reuters.en', 'train_features.tsv')
    data2features('data/test.conll', 'test_features.tsv')
