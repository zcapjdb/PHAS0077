import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix
import random


def load_data(data, columns = [], num_samples = 1, n = 2):

    '''
    Function to load training and testing data in a more memory efficient way by using 8 bit integers
    and control number of columns and rows used

    inputs: data - path to data
            columns - list of columns from data to load
            num_samples - number of rows of training data to load (rows are randomly selected)
            n - size of n-grams used, larger n-grams have more columns to select from
    
    output:
            dataset - returns pandas dataframe of data with required number of columns and rows
    '''      

    if columns != []:
        if isinstance(columns[0], str): # allow for passing of column names or indices
            use_columns = ['file','class'] + columns
        else:
            use_columns = [0,1] + columns
    else:
        # No columns passed, load entire dataset
        use_columns = list(range(256**n + 2))

    # Choose random sample of rows equal to num_samples
    if num_samples != 1:
        total_size = sum(1 for line in open(data)) - 1 # number of data point (excludes header)
        print(total_size)
        sample_size = num_samples
        skip = sorted(random.sample(range(2, total_size + 1), total_size - sample_size)) 
    else:
        skip = 0

    # Use 8-bit integers instead of standard 64-bit integers to save memory when loading dataframe
    data_columns = pd.read_csv(data, nrows=0, usecols = use_columns).columns.tolist()
    dtype_train_dict = {key: 'int8'  for key in data_columns}
    dtype_train_dict['file'] = 'object'
    dataset = pd.read_csv(data, dtype = dtype_train_dict, usecols = use_columns, skiprows = skip )

    return dataset
    

def train(X_train, y_train, X_test,y_test, hidden_layer_sizes = 100):
    '''
    Function to train an n-gram presence model using a neural network on given data and return performance metrics

    inputs: X_train, y_train - training data and training labels
            X_test, y_test - testing data and testing labels
            hidden_layer_sizes - number of neurons in hidden layer, useful to vary when changing number of n-gram inputs
    
    outputs: List containing the accuracy score, precision score, recall score, AUC score and specificity score of model
    
    '''

    clf = MLPClassifier(max_iter=500, random_state=41, hidden_layer_sizes = hidden_layer_sizes)

    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    score_acc = accuracy_score(y_test, y_pred)
    score_prec = precision_score(y_test, y_pred)
    score_rec = recall_score(y_test, y_pred)
    score_auc = roc_auc_score(y_test, y_pred)

    tn, fp, _, _ = confusion_matrix(y_test, y_pred).ravel()
    score_spec = tn / (tn+fp)

    print("Accuracy:", score_acc)
    print("Precision:", score_prec)
    print("Recall:", score_rec)
    print("AUC:", score_auc)
    print("Specificity:", score_spec)
    return [score_acc, score_prec, score_rec, score_auc, score_spec]


if __name__ == "__main__":
    print("Useful Functions")