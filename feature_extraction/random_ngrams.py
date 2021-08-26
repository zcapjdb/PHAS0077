"""
Streamlining of pre-processing carried out in the distinct_ngrams, distinct_ngrams_total, 
information_gain and construct_ngram_presence files

Model performance independent of set of n-grams used, therefore randomly generate number of n-grams desired and extract from files

"""

import os
from pathlib import Path
import random
import pandas as pd
from tqdm import tqdm
import time

ngram_size = 2
ngram_number = 1000 

hex_chars = '0123456789abcdef'
ngram_list = []

file_dir = "../hex_files/"
test_file_list = "../test_files2.csv"  #"../test_files_alex.csv"

# measure how long for preprocessing to run
start_time = time.time()

# Generate list of random ngrams of size specified
for i in range(ngram_number):
    ngram = ''.join(random.choice(hex_chars) for j in range(2*ngram_size))
    ngram_list.append(ngram)

# Extract test file names from csv
with open(test_file_list) as f:
    test_files = []
    name_list = f.read().splitlines()
    for name in name_list:
        test_files.append(name)

print(len(test_files)) 


# Search for ngrams in file and sort into test or train
test_list = []
train_list = []

for root, dirs, files in os.walk(file_dir):
    for file in tqdm(files):
        
        ngram_dict = {}
        class_label = 1 if Path(file).name.split("_")[0]=="VirusShare" else 0
        ngram_dict["file"] = file
        ngram_dict["class"] = class_label

        with open(os.path.join(root, file)) as hex_file:
            hex_string = hex_file.read().replace("\n", "")

        n_grams = []
        for i in range(int(len(hex_string)/2)-1):
            n_grams.append(hex_string[i*2 : (i+ngram_size)*2])
        n_grams = set(n_grams) # only require set of ngrams, frequency not needed

        for ngram in ngram_list:
            if ngram in n_grams:
                ngram_dict[ngram] = 1
                continue
            else:
                ngram_dict[ngram] = 0
        
        if Path(file).name in test_files:
            test_list.append(ngram_dict)
        
        else:
            train_list.append(ngram_dict)

print(len(train_list))  
print(len(test_list))

test_df = pd.DataFrame.from_dict(test_list)
train_df  = pd.DataFrame.from_dict(train_list)

train_df.to_csv("../temp_csvs/train_random_timed_1000.csv", index = False)
test_df.to_csv("../temp_csvs/test_random_timed_1000.csv", index = False)

end_time = time.time()
print("Time taken = ", start_time - end_time)
          


    
