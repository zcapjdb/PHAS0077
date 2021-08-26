#!/bin/bash

echo "Output directory is: $1"

for FILE in ./*
do
    echo $FILE
    xxd -p $FILE > ./$1/${FILE}.txt
done