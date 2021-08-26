"""
Calculate the n-gram presence percentage for the selected number of n-grams and plot the sorted result by class

"""

import pandas as pd
import matplotlib.pyplot as plt
import random
from useful_functions import load_data


# Select num_ngrams and calculate presence percentage from sample_size number of samples
n = 3
num_ngrams = 1000
plot_ngrams = [0,1] + random.sample(range(2,250**n - 2),num_ngrams)
sample_size = 10000

in_file = "3ngram_presence_train2.csv"
dataset = load_data(in_file, plot_ngrams, sample_size)
print("Training Data Loaded")

X_train = dataset.drop('file', axis='columns')

# Count the number of files each n-gram appears in by class and store as a fraction of class size
ngram_dict = {}
total_size_benign = len(X_train[X_train["class"] == 0])
total_size_test = len(X_train[X_train["class"] == 1])
for ngram in range(len(plot_ngrams) - 1):

    num_in_benign = len(X_train[(X_train["class"] == 0) & (X_train[X_train.columns[ngram]] == 1)])
    
    num_in_malware = len(X_train[(X_train["class"] == 1) & (X_train[X_train.columns[ngram]] == 1)])

    # Check incase 0 file length is passed
    if (total_size_benign != 0) or (total_size_test !=0):
        ngram_dict[X_train.columns[ngram]] = [num_in_benign/ total_size_benign, num_in_malware/ total_size_test]
    else:
        ngram_dict[X_train.columns[ngram]] = [0,0]

out_file = "3ngram_presence_percentage.csv"
df = pd.DataFrame.from_dict(ngram_dict, orient = 'columns')
df.to_csv(out_file, index = False)

# Sum the columns for each row, as each n-gram has a 1 or 0 presence this gives the number of n-grams present in file
X_train.loc[:,'Total'] = X_train.iloc[1:].sum(axis=1) #excluding class column
df = [dataset['file'], X_train[['class','Total']]]
df = pd.concat(df, axis = 1)
df.to_csv("3ngram_file_percentages.csv", index = False) 


df = pd.read_csv(out_file)

# Sort n-gram presence from greatest to least present in benign files
benign_list = df.iloc[0,1:].to_list()
malware_list = df.iloc[1,1:].to_list()

idxs = list(range(len(benign_list)))
idxs.sort(key=benign_list.__getitem__) # gets sort idx for benign files

benign_sorted = list(map(benign_list.__getitem__, idxs)) # apply sort idx to benign files and to malicious files below
malware_sorted = list(map(malware_list.__getitem__, idxs))

plt.figure()
plt.plot(benign_sorted[::-1], label = "Benign")
plt.plot(malware_sorted[::-1], label = "Malware")
plt.ylabel("Percentage of files ngram is present in")
plt.xlabel(f"{len(plot_ngrams) - 2}")
plt.legend()
plt.show()
plt.savefig("ngram_presence.png")




