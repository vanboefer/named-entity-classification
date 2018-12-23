import json
import pandas as pd

def feat_dict_list(df, col_idx=[2,3,4,5,6,7,8]):
    """
    Process DataFrame with features and output a list of dictionaries where each dict represents the features of one row (token).
    :param df: DataFrame with columns of features
    :param col_idx: a list of integers that indicate the indices of all columns that contain input features (default=[2,3,4,5,6,7,8])
    :returns: a list of dictionaries where each dict represents the features of one row (token):
        [{'f1_name':'f1_value', 'f2_name':'f2_value', ...}, {'f1_name':'f1_value', 'f2_name':'f2_value', ...}...]
    """
    feat_dict_list = list()
    for i in range(0, len(df)):
        feat_dict = dict()
        for col in col_idx:
            dict_val = df.iloc[i, col]
            dict_key = df.columns.values[col]
            feat_dict[dict_key] = dict_val
        feat_dict_list.append(feat_dict)
    return feat_dict_list


def dict2json(dict_list, outfilepath):
    """
    Write a list of dictionaries to a json file.
    :param dict_list: list of dictionaries
    :param outfilepath: path to the output file
    :returns: None (a json file is created in the indicated directory)
    """
    with open(outfilepath, 'w') as outfile:
        json.dump(dict_list, outfile)
    return None


if __name__ == '__main__':
    # read the features data from the csv files
    train_df = pd.read_csv('train_features.tsv', sep='\t', keep_default_na=False)
    test_df = pd.read_csv('test_features.tsv', sep='\t', keep_default_na=False)

    # create a list of feature dictionaries
    train_feat_dict_list = feat_dict_list(train_df)
    test_feat_dict_list = feat_dict_list(test_df)

    # write the lists to json files
    dict2json(train_feat_dict_list, 'train_X_features.json')
    dict2json(test_feat_dict_list, 'test_X_features.json')
