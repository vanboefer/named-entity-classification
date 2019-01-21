# Named Entity Classification

## Supervised Machine Learning
1. [Repo Structure](https://github.com/vanboefer/named-entity-classification#repo-structure)
2. [Data](https://github.com/vanboefer/named-entity-classification#data)
3. [Features](https://github.com/vanboefer/named-entity-classification#features)
4. [Machine Learning Models](https://github.com/vanboefer/named-entity-classification#machine-learning-models)
5. [Step-by-step Instructions](https://github.com/vanboefer/named-entity-classification#step-by-step-instructions)

---
This repo runs various supervised machine learning algorithms on the task of named entity classification in English. We compare between the results obtained with different models, different features and different settings of hyperparameters. The code generates result reports in `html` format; see example [here](https://vanboefer.github.io/named-entity-classification/class_report_feat2_1.html).

The repo is written in **Python 3**.
- If you installed Python using [Anaconda](https://www.anaconda.com), the only additional dependency you need to install is [sklearn-crfsuite](https://sklearn-crfsuite.readthedocs.io/en/latest/install.html).
- If you did not use Anaconda distribution, make sure you have the following dependencies installed:
    - numpy
    - pandas
    - nltk
    - matplotlib
    - scikit-learn
    - scipy
---
#### REPO STRUCTURE
- **root** directory contains:
    - the Python scripts that you run
    - the xlsx file that defines each experimental run in terms of:
        - the model used (e.g. Logistic Regression)
        - the features used (e.g. feat2)
        - optimization (with / without)
- **models** contains helper functions to run the various ML models
- **utils** contains helper functions to extract features and to create the result reports
- **templates** contains the `html` templates and `css` styles used in the result reports
- **docs** contains example result reports
---
#### DATA

- The data used in the project is in the format of the [CoNLL-2003 shared task](https://www.clips.uantwerpen.be/conll2003/ner/) (see example below). Entities are annotated with LOC (location), ORG (organization), PER (person) and MISC (miscellaneous); non-entities are annotated as O (other). The I- and B- distinction is not used in this project; it is removed during pre-processing. We only use the 'Token' and 'NER' columns.

| Token       | POS | Chunk | NER
|-------------|-----|-------|--------
| U.N.        | NNP | I-NP  |  I-ORG
| official    | NN  | I-NP  |  O
| Ekeus       | NNP | I-NP  |  I-PER
| heads       | VBZ | I-VP  |  O
| for         | IN  | I-PP  |  O
| Baghdad     | NNP | I-NP  |  I-LOC
| .           | .   | O     |  O
- The data is not included in the repo. A free dataset can be downloaded from [here](https://www.clips.uantwerpen.be/conll2003/ner/). **Please note**: the code assumes that training and test data are in two separate files; if you have one file, you need to split it yourself before running the code.
---
#### FEATURES

We experiment with 4 types of features configurations:
- feat0:
    - lemma of the current word (w)
    - lemma of w-1
    - lemma of w+1
- feat1:
    - word embedding of the current word (w)
    - word embedding of w-1
    - word embedding of w+1
- feat2:
    - lemma of the current word (w)
    - lemma of w-1
    - lemma of w+1
    - POS of w
    - POS of w-1
    - POS of w+1
    - word shape: lowcase / upcase_BOS / upcase_IN / all_caps / other
- feat3:
    - word embedding of the current word (w)
    - word embedding of w-1
    - word embedding of w+1
    - POS of w
    - POS of w-1
    - POS of w+1
    - word shape: lowcase / upcase_BOS / upcase_IN / all_caps / other
---
#### MACHINE LEARNING MODELS

We experiment with the following classifiers:
- Logistic Regression
- Naive Bayes (multinomial, complement)
- Linear SVM
- Hidden Markov Models
- Conditional Random Fields
---
#### STEP-BY-STEP INSTRUCTIONS
1) Run ***config.py***; this -
- creates additional directories in the project folder: *data_raw*, *data_processed*, *word_embeddings_EN*, *results*;
- creates the *config.ini* file, which stores all the project paths used in the scripts.

2) Place train and test data (two separate files) in the *data_raw* directory.

3) Download the pre-trained fastText word embeddings from [here](https://fasttext.cc/docs/en/english-vectors.html). Place the `vec` file in the *word_embeddings_EN* directory.

4) Run ***create_X_y.py***. From the menu, specify the training and test file. The script processes the raw data and generates features. The features are stored in `pkl` (pickle) and `npz` (numpy zipped archive) format in the *data_processed* directory. **Please note**: this might take up to 50 minutes.

5) Run ***run_models.py***. This runs the ML models, according to the settings specified in the *ML_configurations.xlsx* file. The predictions of each run are stored in the *results* directory (*predictions.tsv*); the parameter values of each run are also stored in the *results* directory (*best_params.pkl*).**Please note**: this might take up to 80 minutes.

6) Run ***create_results_html.py***. This generates the `html` result reports (in the *results* directory).
