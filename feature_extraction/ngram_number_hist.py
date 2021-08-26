"""
Iterate through dataset and record the total number of n-grams and unique number of n-grams present
Save results to file and plot distributions of both by class
"""


import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv


sub_dirs = ["00352_windows_good_test", "00355_windows_good_test", "00380_windows_good_train", 
"00381_windows_good_test","00387_windows_good_train", "test_benign_dll", "test_benign_exe",
"train_benign_dll1", "train_benign_dll2", "train_benign_exe"]

input_dir = "../XOR"

start_time = time.time()

benign_length = []
malware_length = []

benign_distinct = []
malware_distinct = []

output_file = "../temp_csvs/XOR_3grams.csv"
with open(output_file, "w") as outfile:
    writer = csv.writer(outfile, delimiter=",")
    writer.writerow(["file", "class", "number", "length"])
    for dir in sub_dirs:
        filenames = sorted(os.walk(input_dir + "/" + dir))[0][2]

        print(dir)
        for filename in tqdm(filenames):
            with open(input_dir + "/" + dir + "/"+ filename) as ngram_file:
                file_contents = ngram_file.readlines()
                file_length = len(file_contents)
                file_length_distinct = len(set(file_contents)) # set gets distinct ngrams

                base_name = os.path.splitext(os.path.basename(filename))[0]  # remove path and .txt extension from binary file name
                class_label = 1 if base_name.split("_")[0]=="VirusShare" else 0
                writer.writerow([base_name, class_label, file_length_distinct, file_length])
                
                if base_name.split("_")[0]=="VirusShare":
                    malware_length.append(file_length)
                    malware_distinct.append(file_length_distinct)
                else:
                    benign_length.append(file_length)
                    benign_distinct.append(file_length_distinct)
                    

end_time = time.time()

print("Time taken = ", end_time - start_time)

print(len(malware_length))
print(len(benign_length))
print(max(malware_length))
print(max(benign_length))

plt.figure()
plt.hist([benign_length, malware_length], label=[f'benign, n = {len(benign_length)}', f'malware, n = {len(malware_length)}'], bins = 50)
plt.legend(loc='upper right')
plt.yscale('log', nonpositive='clip')
plt.show()
plt.savefig("../3grams_results/3gram_hist.png")


plt.figure()
plt.hist([benign_distinct, malware_distinct], label=[f'benign, n = {len(benign_length)}', f'malware, n = {len(malware_length)}'], bins = 50)
plt.legend(loc='upper right')
plt.yscale('log', nonpositive='clip')
plt.show()
plt.savefig("../3grams_results/3gram_distinct_hist.png")




