"""
Original Code: Toni Mlinarevic
Edited by: Jackson Barr
Modified to extend chunking method to any n-gram size and control interval size.
Options to change sample size of data used to perform calculation


Perform information gain calculatio to choose n-grams for model

"""


import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random


n = 3   # length of n-gram
# only use training sample to calculate information gain!
ngram_presence_file = "../3gram_presence_2021-07-01_15-51-19.csv"#f"{n}grams/{n}gram_presence_train.csv"

# Perform infogain calculation in chuncks
intervals =  np.arange(2, 256**n, 5000)
intervals = np.append(intervals, 256*n)

for i in tqdm(range(len(intervals) - 1)):
    info_gain_file = "../info_gain/NEW_3gram_info_gain_" + str(intervals[i]) + "-" + str(intervals[i+1]) + "_"  +datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv" #f"{n}grams/{n}gram_info_gain_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

    n = 50000 #number of records in file
    s = 5000 #desired sample size
    skip = sorted(random.sample(range(1,n+1),n-s))

    print("Loading dataset between columns ", intervals[i], " and columns ", intervals[i+1],":")
    data = pd.read_csv(ngram_presence_file, usecols=[1] + [j for j in range(intervals[i],intervals[i+1])], dtype=int, skiprows = skip)
    print("Loaded")

    n_samples = len(data)
    p_benign = len(data[data["class"]==0])/n_samples
    p_malicious = len(data[data["class"]==1])/n_samples
    p_c = [p_benign, p_malicious]
    print("Number of samples:", n_samples)
    print("Fraction of samples benign samples:", p_benign)
    print("Fraction of samples malware samples:", p_malicious)

    IGs = []
    non_zero_p_v_cs = []                  # this is to test why I am getting zero division error

    for column in tqdm(data.columns[1:]):
        data_sub = data[["class", column]]
        IG = 0
        for v in [0,1]:
            p_v = len(data_sub[data_sub[column] == v])/n_samples
            for c in [0, 1]:
                data_c = data_sub[data_sub["class"] == c]
                data_v = data_c[data_c[column] == v]
                p_v_c = len(data_v)/len(data_c)

                if (p_v == 0 or p_c[c] == 0) and p_v_c != 0:    # this is to test why I am getting zero division error
                    # I think this shouldn't be possible, that p_v is 0 but p_v_c is not
                    print(column)
                    print(p_v_c)
                    non_zero_p_v_cs.append(p_v_c)
                    pass
                else:
                    el = 0 if p_v_c == 0 else p_v_c * np.log(p_v_c / p_v / p_c[c])
                    IG += el
                
        IGs.append(IG)

    with open(info_gain_file, "w") as f:
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(data.columns[1:])
        csv_writer.writerow(IGs)

