"""
This script generates an html page with the results for each type of run.

The helper functions for this script are found in the utils package:
- evaluation
"""

import pandas as pd
from utils import evaluation
from config import ProjectPaths


# set paths
path_templates = ProjectPaths()['templates']
path_results = ProjectPaths()['results']

# load predictions from file
predictions_df = pd.read_csv(path_results / 'predictions.tsv', sep='\t')

# load template
template = (path_templates / 'classification_report.html').read_text()

# define y_test (gold labels)
y_test = predictions_df['y_test_GOLD']

# html pages (each item in the list is a type of run (features + optimization)
# the results of all classifiers for this type of run will appear on one page
pages_lst = [
    'feat0_0',
    'feat0_1',
    'feat1_0',
    'feat2_0',
    'feat2_1',
    'feat3_0',
]
# generate a page for each type of run
# (with the results of all classifiers for this type of run)
for page in pages_lst:
    toc = ''
    block = ''
    for col in predictions_df.columns:
        if page in col:
            y_pred = predictions_df[col]
            block += evaluation.results2html(y_test, y_pred)
            section_name = evaluation.name2title(y_pred.name).split(',')[0]
            toc += f'<a href="#{y_pred.name}"><div class="button">{section_name}</div></a>\n'
    temp_out = template.replace('[REPORT_TITLE]', page)
    temp_out = temp_out.replace('[TOC]', toc)
    temp_out = temp_out.replace('[REPORT_BLOCK]', block)

    (path_results / ('class_report_' + page + '.html')).write_text(temp_out)
