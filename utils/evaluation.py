import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from config import ProjectPaths
path_results = ProjectPaths()['results']
path_templates = ProjectPaths()['templates']


def results2html(y_test, y_pred):
    """
    tbd
    """

    # generate the description of the report
    description = name2title(y_pred.name) + f'{get_param_val(y_pred.name)}'

    # generate the report
    clasrep_dict = classification_report(y_test, y_pred, output_dict=True)
    df_rep = pd.DataFrame(clasrep_dict).T
    df_rep['support'] = df_rep['support'].astype(int)
    clasrep = df_rep.round(2).to_html()

    # generate confusion matrix plots
    path_results = ProjectPaths()['results']
    plt_norm = cm_plot(y_test, y_pred, normalize=True)
    plt_no_norm = cm_plot(y_test, y_pred, normalize=False)

    # load template
    report_block = (path_templates / 'report_block.html').read_text()
    # populate template
    block_vars = {
        '[ID]': y_pred.name,
        '[DESCRIPTION]': description,
        '[TABLE]': clasrep,
        '[CM_NORM]': plt_norm,
        '[CM_NO_NORM]': plt_no_norm,
        }
    for var in block_vars:
        report_block = report_block.replace(var, block_vars[var])

    return report_block

def name2title(run_name):
    """
    tbd
    """

    model_code, feat, optimized = run_name.split('_')

    model_dict = {
        'mnb': 'Multinomial Naive Bayes',
        'cnb': 'Complement Naive Bayes',
        'crf': 'Conditional Random Field',
        'hmm': 'Hidden Markov Model',
        'lr': 'Logistic Regression',
        'lsvc': 'Linear SVM',
    }

    optim_dict = {
        '0': 'without optimization',
        '1': 'with optimization',
    }

    return f'{model_dict[model_code]}, {feat}, {optim_dict[optimized]}'


def get_param_val(run_name):
    """
    tbd
    """

    with open(path_results / 'best_params.pkl', 'rb') as infile:
        params = pickle.load(infile)

    param_val = params[run_name]
    param_str = ''
    try:
        for key in param_val:
            param_str = ' '.join([param_str, f'{key}={param_val[key]:.3f}'])
        param_str = f' ({param_str} )'
    except:
        param_str = ''

    return param_str


def cm_plot(y_test, y_pred, normalize=False):
    """
    Based on: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    """

    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    title = y_pred.name + ' not normalized'
    plt.title('Not normalized')
    class_names = ['LOC', 'MISC', 'O', 'ORG', 'PER']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = y_pred.name + ' normalized'
        plt.title('Normalized')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    path_to_save = path_results / 'png' / (title + '.png')
    plt.savefig(path_to_save, dpi=150)
    plt.close()

    return title
