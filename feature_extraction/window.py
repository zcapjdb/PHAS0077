"""
Train a large number of models across a sliding window of n-gram presence curve
Record performance as function of n-gram presence delta

"""

from numpy.lib.npyio import load
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from useful_functions import load_data, train

# Load n-gram presence percentage data and sort by biggest presence delta between classes
df = pd.read_csv("temp_csvs/ngram_presence_percentage_full.csv")
df_columns = np.asarray(df.columns.to_list())

difference_list = abs(np.asarray(df.iloc[0,:].to_list()[1:]) - np.asarray(df.iloc[1,:].to_list()[1:]))
sorted = np.argsort(difference_list)[::-1]
sorted_difference = difference_list[sorted]

df_columns_sorted = df_columns[sorted]
print(df_columns_sorted)

# Set size of models that will be trained
model_size = 5
print(df_columns[sorted[0]: sorted[0] + model_size])


training_data = "ngram_presence_train.csv"
testing_data = "ngram_presence_test.csv"

dataset = load_data(training_data)
print("Loaded Training Data")

dataset_test = load_data(testing_data)
print("Loaded Testing Data")


y_train = dataset['class']
y_test = dataset_test['class']

acc_list, prec_list, rec_list, auc_list, spec_list = [], [], [], [], []
presence_delta = [] 
for i in tqdm(range(len(df_columns_sorted[::model_size]) - 1)):

    # Creates list of ngrams of size model_size with consecutive presence delta values
    columns = df_columns_sorted[i * model_size : (i + 1) * model_size]
    print(columns)

    presence_delta.append(i)

    X_train = dataset[columns]
    X_test = dataset_test[columns]

    score_acc, score_prec, score_rec, score_auc, score_spec = train(X_train, y_train, X_test, y_test)

    acc_list.append(score_acc)
    prec_list.append(score_prec)
    rec_list.append(score_rec)
    auc_list.append(score_auc)
    spec_list.append(score_spec)


# Store for future analysis
df = pd.DataFrame(list(zip(presence_delta, acc_list, prec_list, rec_list, spec_list, auc_list)), 
                  columns = ['Presence Delta','Accuracy', 'Precision','Recall', 'Specificity','Area under Curve'])
df.to_csv("window_full.csv", index = False)

'''
print("Loading ngram results")
df = pd.read_csv("temp_csvs/window_full.csv")
presence_delta = df['Presence Delta'] 
acc_list = df['Accuracy']
prec_list = df['Precision']
rec_list = df['Recall']
spec_list = df['Specificity']
auc_list = df['Area under Curve']
'''

# Calculate rolling average as results have large amount of noise
acc_rolling = df['Accuracy'].rolling(window=50).mean()
prec_rolling = df['Precision'].rolling(window=50).mean()
rec_rolling = df['Recall'].rolling(window=50).mean()
auc_rolling = df['Area under Curve'].rolling(window=50).mean()
spec_rolling = df['Specificity'].rolling(window=50).mean()

plt.figure()
plt.plot(presence_delta, acc_rolling, label = 'Accuracy', color = 'red')
plt.plot(presence_delta, prec_rolling, label = 'Precision', color = 'blue')
plt.plot(presence_delta, rec_rolling, label = 'Recall', color = 'green')
plt.plot(presence_delta, auc_rolling, label = 'AUC', color = 'orange')
plt.plot(presence_delta, spec_rolling, label = 'Specificity', color = 'magenta')

plt.xlabel("Presence Delta Rolling Average")
plt.ylabel("Performance Metrics Score")
plt.legend(loc='center left', bbox_to_anchor=(0, 0.70),
          ncol=1, fancybox=True, shadow=True)
plt.show()

#plt.savefig("presence_delta_window_rolling.pdf")
#plt.savefig("presence_delta_window_rolling.svg")
plt.savefig("presence_delta_window_rolling.png")