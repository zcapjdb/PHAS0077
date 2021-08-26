"""
Speed test of different methdos of storing n-grams, dump to text file or dump to csv using pandas or numpy

"""
from distinct_ngrams import extract_ngrams
import timeit

input_dir = "../hex_files/test_benign_dll"
output_dir = "speed_test"

files = ["B.dll.txt", "B.xs.dll.txt", "B330.dll.txt", "BackgroundActivatorDLL.dll.txt","Battery.dll.txt",
        "BIBUtils.dll.txt","BitbucKitServer.Net.Api.dll.txt","BrNetToolAru.dll.txt","eapacfg.dll.txt","exp.dll.txt"]

def time_func(files, input_dir, output_dir, write):
    for file in files:
        extract_ngrams(file, input_dir, output_dir, write)

panda_times = timeit.Timer(stmt = "speed_test.time_func(speed_test.files,speed_test.input_dir, speed_test.output_dir, 'panda')",
 setup = "import speed_test" ).repeat(repeat = 10, number = 10)

csv_times = timeit.Timer(stmt = 'speed_test.time_func(speed_test.files, speed_test.input_dir, speed_test.output_dir, "csv")',
 setup = 'import speed_test').repeat(repeat = 10, number = 10)

np_times = timeit.Timer(stmt = 'speed_test.time_func(speed_test.files, speed_test.input_dir, speed_test.output_dir, "numpy")',
setup = 'import speed_test').repeat(repeat = 10, number = 10)

print(panda_times)
print(csv_times)
print(np_times)

print("Min panda time: ", min(panda_times))
print("Min csv time:", min(csv_times) )
print("Min numpy time:", min(np_times) )
    