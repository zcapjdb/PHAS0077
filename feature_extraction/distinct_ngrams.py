"""
Original Code: Toni Mlinarevic
Edited by: Jackson Barr
Modified to include timeout if ngram extraction takes too long / gets stuck
and wrapped extraction process in a function


For each file in input directory (which contains hexadecimal representations of executables),
find unique n-grams and save them in a new file of the same name in output directory

"""

import os
from tqdm import tqdm

from functools import wraps
import errno
import signal
from pathlib import Path
import pandas as pd
from numpy import savetxt

class TimeoutError(Exception):
    pass

# Decorator that uses signal library to raise error if ngram extraction takes longer than specified time
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(20) # allows 20 seconds per file
def extract_ngrams(file, input_dir, output_dir, write = "text"):
    with open(input_dir +"/"+ file) as hex_file:
            hex_string = hex_file.read().replace("\n", "")

            n_grams = []
            for i in range(int(len(hex_string)/2)-1):
                n_grams.append(hex_string[i*2 : (i+n)*2])

            # Options to choose how data is stored
            if write == "panda":
                df = pd.DataFrame(n_grams)
                df.to_csv(output_dir  + "/" + file, index = False)
            elif write == "text":
                fileout = open(output_dir  + "/" + file,"w")
                for n_gram in n_grams:
                    fileout.write(n_gram + "\n")
                fileout.close()
            elif write == "numpy":
                filename = output_dir  + "/" + file
                savetxt(filename, n_grams, newline = "\n", fmt = "%s")



sub_dirs = ["00352_windows_good_test", "00355_windows_good_test", "00380_windows_good_train", 
"00381_windows_good_test","00387_windows_good_train", "test_benign_dll", "test_benign_exe",
 "train_benign_dll1", "train_benign_dll2", "train_benign_exe"]


input_dir = "../XOR"
output_dir = "../hex_distinct/XOR_2grams"
n = 2 # n-gram size

if __name__ == "__main__":

    broken_files = []
    for dir in sub_dirs:
        filenames = sorted(os.walk(input_dir + "/" + dir))[0][2]

        if not os.path.exists(output_dir + "/" + dir):
            os.makedirs(output_dir + "/" + dir)

        print(dir)
        for filename in tqdm(filenames):
            in_dir = input_dir + "/" + dir
            out_dir = output_dir + "/" + dir
            try: 
                extract_ngrams(filename, in_dir, out_dir)
            except Exception:
                broken_files.append(in_dir + "/" + filename)
                print("Skipping: " + in_dir + "/" + filename)

    df = pd.DataFrame(broken_files)
    df.to_csv("broken_files.csv", index = False)
