"""
Create list of test files from directories with test in name
"""


import os
import pandas as pd
from pathlib import Path

sub_dirs = ["00352_windows_good_test", "00355_windows_good_test", "00380_windows_good_train", 
"00381_windows_good_test","00387_windows_good_train", "test_benign_dll", "test_benign_exe",
 "train_benign_dll1", "train_benign_dll2", "train_benign_exe"]

files = []
for dir in sub_dirs:
    if "test" in dir:
        files += ["../hex_distinct/" + dir + "/" + file for file in sorted(os.walk("../hex_distinct/" +dir))[0][2]]

files_clean = []
for file in files:
    files_clean.append(Path(file).name)

output_file = "../test_files2.csv"
df = pd.DataFrame(files_clean)
df.to_csv(output_file, index=False)