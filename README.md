# PHAS0077

Code used as part of PHAS0077 project:
https://github.com/sebastien-rettie/cdt-nccgroup used as starting code base.

Note: Due to malicious nature of files used throughout this project and benign files sourced from personal machines the required data is not included in this repository. Data is stored on the AWS virtual machines tests were conducted on.


Standard Workflow:
- distinct_ngrams.py
- distinct_ngrams_total.py
- construct_ngram_presence.py
- train_test_split.py
- information_gain.py
- ngrams_models.ipynb

Starting with the hex representations of files, this workflow generates the list of n-grams in each file, determines the set of n-grams across all files, constructs the full feature vectors for each file. The data is then split into the training and testing sets and information gain is applied to the training set to reduce the number of features fed into the n-gram presence model. This workflow is unchanged from the starting code base, with some optmisations made throughout.

Alternate workflow:
- random_ngrams.py
- ngrams_models.ipynb

This alternate workflow combines all the pre-processing to one step. By leaving out the information gain feature selection and randomly selecting n-grams, only the n-grams used in the model need to be searched for instead of constructing the presence of all n-grams and then reducing the dimensionality.

n-gram distribution model workflow:
- ngram_number_hist.py 
- Distribution_models.ipynb

The ngram_number_hist.py file generates the number of n-grams (total number and unique number) and saves to file that can be loaded in the notebook. Combined model of n-gram presence and n-gram distribution also contained in this notebook.

Other files contain other tests of models and dataset visualisation, look at file docstrings for details of specific files. The useful_functions.py containes a couple frequently used functions used across the repository.
