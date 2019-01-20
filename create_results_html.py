import pandas as pd
from utils import evaluation
from config import ProjectPaths
path_templates = ProjectPaths()['templates']
path_results = ProjectPaths()['results']

# load predictions from file
predictions_df = pd.read_csv(path_results / 'predictions.tsv', sep='\t')

pages_lst = [
    'feat0_0',
    'feat0_1',
    'feat1_0',
    'feat2_0',
    'feat2_1',
    'feat3_0',
]

template = (path_templates / 'classification_report.html').read_text()

y_test = predictions_df['y_test_GOLD']

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
