import os
from tqdm import tqdm
import time

def genkey(length):
    return os.urandom(length)

def xor_strings(s, t):
    """xor two strings together."""
    if isinstance(s, str):

        return b"".join(chr(ord(a) ^ ord(b)) for a, b in zip(s, t))
    else:
        return bytes([a ^ b for a, b in zip(s, t)])


sub_dirs = ["test_benign_dll", "test_benign_exe",
"train_benign_dll1", "train_benign_dll2", "train_benign_exe"]

input_dir = "hex_files"
output_dir = "XOR"

distinct_n_grams = set()

start_time = time.time()

# Iterate through all benign files, xor cipher each and save to new file in output directory
for dir in sub_dirs:
    filenames = sorted(os.walk(input_dir + "/" + dir))[0][2]

    if not os.path.exists(output_dir + "/" + dir):
        os.makedirs(output_dir + "/" + dir)

    print(dir)
    for filename in tqdm(filenames):
        with open(input_dir + "/" + dir + "/"+ filename) as hex_file:
            hex_string = hex_file.read().replace("\n", "")
            key = genkey(len(hex_string))

            hex_string = bytes.fromhex(hex_string)
            cipher_text = xor_strings(hex_string, key)           
            cipher_text_hex = cipher_text.hex()
        
        with open(output_dir  + "/" + dir + "/" + filename,"w") as out_file:
            out_file.write(cipher_text_hex)

end_time = time.time()
print("Time taken = ", end_time - start_time)