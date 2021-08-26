"""
Performs a user set number of trials on models with a small number of inputs selected from subset of n-grams
Record performance of each model and which n-grams were used
Record n-grams that appear much higher than expected in best performing models and plot them on presence curve
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

from collections import Counter
import json
from itertools import dropwhile
from useful_functions import load_data, train 


training_data = "temp_csvs/ngram_presence_train.csv"
testing_data = "temp_csvs/ngram_presence_test.csv"
num_trials = 10000

# Set of n-grams used in test
n = 2
size = 1000
ngrams_used = [0,1] + random.sample(range(2,250 ** n - 2),size)

print("Loading Training Data")
dataset = load_data(data = training_data, columns = ngrams_used)
print("Training Data Loaded")

print("Loading Testing Data")
dataset_test = load_data(data = testing_data, columns = ngrams_used)
print("Testing Data Loaded")

X_train = dataset.drop('file', axis='columns')
X_train.drop('class', axis = 'columns')
y_train = dataset['class']


X_test = dataset_test.drop('file', axis='columns')
X_test.drop('class', axis = 'columns' )
y_test = dataset_test['class']

# Plan: Train lots of classifiers with only a few ngrams 
acc_list, prec_list, rec_list, auc_list, spec_list = [], [], [], [], []
ngrams_used = []
for i in tqdm(range(num_trials)): 

    # Choose 5 n-grams from set of n_grams and store
    model_size = 5
    random_vals = random.sample(range(2, size - 1), model_size)
    col_names = []
    for random_val in random_vals:
        col_names.append(X_train.columns[random_val])

    X_train_iter = X_train[col_names]
    X_test_iter = X_test[col_names]

    ngrams_used.append(X_test_iter.columns.tolist())

    y_train_iter = y_train #test only has one column
    y_test_iter = y_test

    score_acc, score_prec, score_rec, score_auc, score_spec = train(X_train_iter, y_train_iter, X_test_iter, y_test_iter)

    acc_list.append(score_acc)
    prec_list.append(score_prec)
    rec_list.append(score_rec)
    auc_list.append(score_auc)
    spec_list.append(score_spec)

df = pd.DataFrame(list(zip(ngrams_used,acc_list, prec_list, rec_list, spec_list, auc_list)), 
                  columns = ['Ngrams','Accuracy', 'Precision','Recall', 'Specificity','Area under Curve'])
df.to_csv("small_ngrams.csv", index = False)


#print("Loading ngram results")
#df = pd.read_csv("small_ngrams.csv")


# Histogram of performance metrics
df.drop('Ngrams', axis = 1)
df.hist(sharex = True, sharey = True)
plt.savefig("histograms.png")

# Get list of n-grams from each test
def flatten_list(df, fraction, num_trials, sort_metric = 'Accuracy'):
    df = df.sort_values(by=[sort_metric], ascending = False)
    top_ngram_lists = df['Ngrams'].head(int(num_trials * fraction))

    # top_ngrams_list is a string of nested  lists, need to convert to list
    flat_list = []
    for sub_list in top_ngram_lists:
        new_list = sub_list.replace('\'', '"') # need to replace the single quotes with doubles or json load doesn't work
        jsonified = json.loads(new_list)
        for element in jsonified:     
            flat_list.append(element)
    
    return flat_list

# Total number of times each n-gram is selected
flat_list_full = flatten_list(df, 1, num_trials)
counted_full = Counter(flat_list_full)

plt.figure()
dict_full = list(counted_full.values())
plt.hist(dict_full, bins = 15)
plt.xlabel("Number of times ngram is selected")
plt.ylabel("Frequency")
plt.savefig("selected_hist.png")


# 10000 trials with 5 ngrams each using a sample of 1000 ngrams means each ngram should be selected on average 50 times
# in the top 10% of trials each ngram should only appear an expected 5 times
# plot ngrams that exceed significantly exceed that amount
flat_list_best = flatten_list(df, 0.1, num_trials) 
counted_dict = Counter(flat_list_best) # gives the number of times n-gram appears in top 10% of models
print(counted_dict.most_common(50))

plt.figure()
dict_values = list(counted_dict.values())
plt.hist(dict_values, bins = 15)
plt.xlabel("Number of times ngram is selected in best models")
plt.ylabel("Frequency")
plt.savefig("selected_best_hist.png")

with open('count.txt', 'w') as file:
     file.write(json.dumps(counted_dict)) # use `json.loads` to do the reverse

for key, count in dropwhile(lambda key_count: key_count[1] >= 20, counted_dict.most_common()): # get only n-grams with greater than 20 appearances
    del counted_dict[key]

print(counted_dict)
most_common_keys = list(counted_dict.keys())


print("Beginning data loading for most common keys")
dataset = load_data(data = training_data, columns = most_common_keys)
dataset_test = load_data(data = testing_data, columns = most_common_keys)
print("Testing Data Loaded")
X_train = dataset.drop('file', axis='columns')
X_test = dataset_test.drop('file', axis='columns')

# Get presence percentages of most commonly appearing n-grams
ngram_presence = {} 
total_size_benign = len(X_train[X_train["class"] == 0])
total_size_test = len(X_train[X_train["class"] == 1])
for ngram in most_common_keys:

    num_in_benign = len(X_train[(X_train["class"] == 0) & (X_train[ngram] == 1)])
    num_in_malware = len(X_train[(X_train["class"] == 1) & (X_train[ngram] == 1)])
    ngram_presence[ngram] = [num_in_benign/ total_size_benign, num_in_malware/ total_size_test]


df1 = pd.DataFrame.from_dict(ngram_presence, orient = 'columns')
# plot presence percentage graph from small_ngrams.py and see where these most common ngrams fall
df2 = pd.read_csv("temp_csvs/ngram_presence_percentage_full.csv", index_col = False)

df = pd.concat([df1,df2], axis = 1).drop_duplicates().reset_index(drop=True)

benign_list = df.iloc[0,:].to_list()
malware_list = df.iloc[1,:].to_list()

idxs = list(range(len(benign_list)))
idxs.sort(key=benign_list.__getitem__)

benign_sorted = list(map(benign_list.__getitem__, idxs))[::-1] # put in descending order
malware_sorted = list(map(malware_list.__getitem__, idxs))[::-1]


best_benign_list = df1.iloc[0,:].to_list()
best_malware_list = df1.iloc[1,:].to_list()
intersect, ind_a, ind_b = np.intersect1d(np.asarray(best_benign_list[1:]),np.asarray(benign_sorted), return_indices=True)
# ind_b gives the index values in the sorted list of the best performing ngrams

plt.figure()

# Plots vertical line at most commonly appearing n-gram locations with decreasing opacity for lower number of appearances
i = 0
for x_coord in ind_b:
    if i == 0:
        plt.axvline(x = x_coord, ymin = 0, ymax = 1, color = 'red', alpha = (1 - i), label = "Best 2-grams locations")
    else:
        plt.axvline(x = x_coord, ymin = 0, ymax = 1, color = 'red', alpha = (1 - i))   
    i += 1 / len(best_benign_list)

plt.plot(benign_sorted[:-1], label = "Benign") 
plt.plot(malware_sorted[:-1], label = "Malware")
plt.show()
plt.ylabel("Percentage of files ngram is present in")
plt.xlabel("sorted ngrams presence")

plt.legend(loc = "lower left", bbox_to_anchor=(0, 1),
          ncol=3, fancybox=True, shadow=True)
plt.savefig("best_ngrams_position.png")

