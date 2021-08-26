"""
Create file which has unique ngrams from all files in input directory.
The input directory is the output directory of distinct_ngrams.py.
"""

import os
from tqdm import tqdm
import time

sub_dirs = ["00352_windows_good_test", "00355_windows_good_test", "00380_windows_good_train", 
"00381_windows_good_test","00387_windows_good_train", "test_benign_dll", "test_benign_exe",
"train_benign_dll1", "train_benign_dll2", "train_benign_exe"]

input_dir = "../hex_distinct/3grams"
output_file = "../distinct_3grams_total.txt"

distinct_n_grams = set()

start_time = time.time()

for dir in sub_dirs:
    filenames = sorted(os.walk(input_dir + "/" + dir))[0][2]

    print(dir)
    for filename in tqdm(filenames):
        with open(input_dir + "/" + dir + "/"+ filename) as ngram_file:
            file_n_grams = ngram_file.readlines()

        distinct_n_grams.update(set(file_n_grams))

with open(output_file,"w") as fileout:
    for n_gram in distinct_n_grams:
        fileout.write(n_gram)

end_time = time.time()
print("Time taken = ", end_time - start_time)