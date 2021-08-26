import csv
from tqdm import tqdm
from pathlib import Path

n = 2   # length of n-gram
ngram_presence_file =  "../3gram_presence_2021-07-01_15-51-19.csv" #"../ngram_presence_2021-06-16_15-07-24.csv"
test_list_file = "../test_files/test_files2.csv"

outfile_train = "../3ngram_presence_train2.csv"#f"{n}grams/{n}gram_presence_train.csv"
outfile_test = "../3ngram_presence_test2.csv"#f"{n}grams/{n}gram_presence_test.csv"

def get_basenames(name_list):  # "base name" here means the name without the path and extension
    return [Path(file_name).stem.split('.')[0] for file_name in name_list] #two .stem as some files have two extensions

with open(test_list_file) as f:
    test_files = get_basenames(f.read().splitlines())


with open(ngram_presence_file) as f:
    fileout_train = open(outfile_train, "w")
    fileout_test = open(outfile_test, "w")
    writer_train = csv.writer(fileout_train, delimiter=",")
    writer_test = csv.writer(fileout_test, delimiter=",")

    csv_reader = csv.reader(f)
    header = next(csv_reader)
    writer_train.writerow(header)
    writer_test.writerow(header)

    counter = 0
    for row in tqdm(csv_reader):
        if Path(row[0]).name in test_files: 
            counter += 1
            writer_test.writerow(row)
        else:
            writer_train.writerow(row)
    
    print(counter)
    fileout_train.close()
    fileout_test.close()