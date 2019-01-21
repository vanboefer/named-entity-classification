"""
This script runs the ML models, according to the settings specified in the ML_configurations.xlsx file. Each row in the table defines a run in terms of:
    - the model used (e.g. Logistic Regression)
    - the features used (e.g. feat2), incl. the relevant filenames
    - optimization (with / without)

The helper functions for this script are found in the models package.
"""

import timeit
import pickle
import pandas as pd

from config import ProjectPaths

import models.CRF
import models.HMM
import models.LogisticRegression
import models.NaiveBayes
import models.SVM


path_proc = ProjectPaths()['data_processed']
path_results = ProjectPaths()['results']
path_cwd = ProjectPaths()['work_dir']

# create DataFrame with the gold NER tags (test data);
# the predictions of the classifiers will be added to this df
with open(path_proc / 'y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

results_df = pd.DataFrame(y_test, columns=['y_test_GOLD'])

# load all the configurations to be run
ml_configs = pd.read_excel(path_cwd / 'ML_configurations.xlsx')

# models
classifiers = {
    'mnb': [models.NaiveBayes.mnb, 0],
    'cnb': [models.NaiveBayes.cnb, 0],
    'lr': [models.LogisticRegression.log_reg, 0],
    'lsvc': [models.SVM.lsvc, 0],
    'crf': [models.CRF.crf, 1],
    'hmm': [models.HMM.supervised_hmm, 2],
}

best_params_dict = dict()

for conf in ml_configs.itertuples():
    start = timeit.default_timer()
    col_name = '_'.join([conf.Model, conf.Features, str(conf.Optimized)])
    print(f'Running {col_name}...')

    arg1 = path_proc / conf.arg1
    arg2 = path_proc / conf.arg2
    try:
        arg3 = path_proc / conf.arg3
    except:
        pass
    opt = bool(conf.Optimized)

    classifier, conf_type = classifiers[conf.Model]
    if conf_type == 0:
        best_params, y_pred = classifier(arg1, arg2, arg3, opt)
    elif conf_type == 1:
        best_params, y_pred_crf, y_pred = classifier(arg1, arg2, opt)
    else:
        y_pred = classifier(arg1, arg2)
        best_params = None

    # store y_pred
    results_df[col_name] = y_pred

    # store best_params
    best_params_dict[col_name] = best_params

    stop = timeit.default_timer()
    ex_time = stop - start
    print(f'{col_name} finished! It took {ex_time/60:.2f} minutes.')

#######################################################
# WRITE RESULTS TO FILES
#######################################################
outfile = path_results / 'predictions.tsv'
results_df.to_csv(outfile, sep='\t')

outfile = path_results / 'best_params.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(best_params_dict, f)
