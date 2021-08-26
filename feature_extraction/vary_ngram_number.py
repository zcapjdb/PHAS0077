"""
Creates a series of n-gram presence models for different numbers of input n-grams repeated a user set amount of times
Calculates the average and standard deviation of performance metrics and plots results

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from useful_functions import train


def chunk_load(data, columns, rows = 10000, chunk_size = 1000):
    '''
    Adaption of load_data function from useful_functions.py
    Loads data in chunks and converts to sparse arrays to save memory
    Set chunk size depending on machine capability
    
    '''

    data_columns = pd.read_csv(data, nrows=0, usecols = columns).columns.tolist()
    dtype_dict = {key: 'int8'  for key in data_columns}
    dtype_dict['file'] = 'object'
    print("Generated Data Types")

    chunk_list = []
    for chunks in pd.read_csv(data , usecols = columns, dtype = dtype_dict, chunksize = chunk_size, nrows = rows):
        chunks = chunks.astype(pd.SparseDtype("int8", np.int8(0)))
        chunk_list.append(chunks)
    
    dataset = pd.concat(chunk_list, axis=0)
    del chunk_list
    print("Loaded Data")

    return dataset


training_data = "temp_csvs/3ngram_presence_train2.csv"
testing_data = "temp_csvs/3ngram_presence_test2.csv"

n = 3
number_of_runs = 10
acc_dict, prec_dict, rec_dict, auc_dict, spec_dict= {}, {}, {}, {}, {}
for j in tqdm(range(number_of_runs)):
    print("Performing run number ", j + 1, ":")

    # Number of input features to be trialed
    ngram_numbers = [100,500,1000,5000,10000,20000,50000,100000]

    train_list = []
    test_list = []

    # Create random list of columns to be used of length equal to maximum number of n-grams tested
    random_list = [1] + random.sample(range(2, 256**n - 1), ngram_numbers[-1]) #not in file name when making sparse array

    chunk_size = 500

    dataset_max = chunk_load(training_data, columns = random_list, rows = 10000, chunk_size  = chunk_size)
    dataset_test_max = chunk_load(testing_data, columns = random_list, rows = 10000, chunk_size  = chunk_size)

    
    # Create subsets of data columns for specific model required
    for i in ngram_numbers:

        if i == ngram_numbers[-1]:
            dataset = dataset_max
            dataset_test = dataset_test_max
        else:
            random_idx = [0,1] + random.sample(range(2, ngram_numbers[-1]-1), i)
            dataset = dataset_max.iloc[:, random_idx]
            dataset_test = dataset_test_max.iloc[:, random_idx]

        X_train = dataset.drop('class', axis='columns')
        y_train = dataset['class']

        X_test = dataset_test.drop('class', axis='columns')
        y_test = dataset_test['class']

        train_list.append((X_train, y_train))
        test_list.append((X_test, y_test))
        
    print("Data Loaded")
    print(len(y_train))
    

    acc_list, prec_list, rec_list, auc_list, spec_list = [], [], [], [], []
    for i in range(len(train_list)):

        print("Using ", ngram_numbers[i], " ngrams")
        score_acc, score_prec, score_rec, score_auc, score_spec = train(train_list[i][0], train_list[i][1],test_list[i][0],
                                                                     test_list[i][1], hidden_layer_sizes = int((2/3)*ngram_numbers[i]) )

        acc_list.append(score_acc)
        prec_list.append(score_prec)
        rec_list.append(score_rec)
        auc_list.append(score_auc)
        spec_list.append(score_spec)


    acc_dict[j] = acc_list
    prec_dict[j] = prec_list
    rec_dict[j] = rec_list
    auc_dict[j] = auc_list
    spec_dict[j] = spec_list

# Store results of runs in csvs for easy future loading
for idx, dict in enumerate([acc_dict, prec_dict, rec_dict, auc_dict, spec_dict]):
    names = ["acc", "prec", 'rec', 'auc', 'spec']
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(f'temp_csvs/{n}gram_{names[idx]}_{number_of_runs}.csv')


# Loading of examples previoulsy ran
acc_array = pd.read_csv("temp_csvs/3gramacc10.csv", header = None).drop(0, axis = 1).drop(0, axis = 0).to_numpy()
prec_array = pd.read_csv("temp_csvs/3gramprec10.csv", header = None).drop(0, axis = 1).drop(0, axis = 0).to_numpy()
rec_array = pd.read_csv("temp_csvs/3gramrec10.csv", header = None).drop(0, axis = 1).drop(0, axis = 0).to_numpy()
auc_array = pd.read_csv("temp_csvs/3gramauc10.csv", header = None).drop(0, axis = 1).drop(0, axis = 0).to_numpy()
spec_array = pd.read_csv("temp_csvs/3gramspec10.csv", header = None).drop(0, axis = 1).drop(0, axis = 0).to_numpy()


def average_metric(metric_array, ngram_numbers = [1,10,50,100,250,500,1000,1500,2000,2500]):
    """
    Calculate the mean and standard deviation of performance metrics at each input n-gram value
    """

    average_list = []
    deviation_list = []
    for j in range(len(ngram_numbers)):
        
        if len(metric_array[j]) == 1:
            average_list.append(metric_array[j].mean())
            deviation_list.append(0)

        else:
            average_list.append(metric_array[j].mean())
            deviation_list.append(metric_array[j].std())


    return np.asarray(average_list), np.asarray(deviation_list)

average_acc, deviation_acc = average_metric(acc_array, ngram_numbers)
average_prec, deviation_prec = average_metric(prec_array, ngram_numbers)
average_rec, deviation_rec = average_metric(rec_array, ngram_numbers)
average_auc, deviation_auc = average_metric(auc_array, ngram_numbers)
average_spec, deviation_spec = average_metric(spec_array, ngram_numbers)


# Plot performance metrics against number of n-grams with mean shown as solid line and standard deviation and standard error on
# the mean shown as shaded regions
root_n = np.sqrt(len(ngram_numbers))
plt.figure()
plt.plot(ngram_numbers, average_acc, label= "Accuracy", color = 'red', linewidth = 0.5)
plt.fill_between(ngram_numbers, average_acc - deviation_acc, average_acc + deviation_acc, alpha = 0.125, color = 'red')
plt.fill_between(ngram_numbers, average_acc - deviation_acc/root_n, average_acc + deviation_acc/root_n, alpha = 0.25, color = 'red')

plt.plot(ngram_numbers, average_prec, label = "Precision", color = 'blue', linewidth = 0.5)
plt.fill_between(ngram_numbers, average_prec - deviation_prec, average_prec + deviation_prec, alpha = 0.125, color = 'blue')
plt.fill_between(ngram_numbers, average_prec - deviation_prec/root_n, average_prec + deviation_prec/root_n, alpha = 0.25, color = 'blue')

plt.plot(ngram_numbers, average_rec, label = "Recall", color = 'green', linewidth = 0.5)
plt.fill_between(ngram_numbers, average_rec - deviation_rec, average_rec + deviation_rec, alpha = 0.125, color = 'green')
plt.fill_between(ngram_numbers, average_rec - deviation_rec/root_n, average_rec + deviation_rec/root_n, alpha = 0.25, color = 'green')

plt.plot(ngram_numbers, average_auc, label = "AUC", color = 'orange', linewidth = 0.5)
plt.fill_between(ngram_numbers, average_auc- deviation_auc, average_auc + deviation_auc, alpha = 0.125, color = 'orange')
plt.fill_between(ngram_numbers, average_auc- deviation_auc/root_n, average_auc + deviation_auc/root_n, alpha = 0.25, color = 'orange')

plt.plot(ngram_numbers, average_spec, label = "Specificity", color = 'magenta', linewidth = 0.5)
plt.fill_between(ngram_numbers, average_spec- deviation_spec, average_spec + deviation_spec, alpha = 0.125, color = 'magenta')
plt.fill_between(ngram_numbers, average_spec- deviation_spec/root_n, average_spec + deviation_spec/root_n, alpha = 0.25, color = 'magenta')

plt.xlabel("Number of ngrams")
plt.ylim([0.75, 1])
plt.legend()
plt.show()

plt.savefig("3grams_results/vary_ngrams10.pdf")
plt.savefig("3grams_results/vary_ngrams10.svg")
plt.savefig("3grams_results/vary_ngrams10.png")
