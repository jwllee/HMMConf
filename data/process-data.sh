#!/usr/bin/env bash 
wget "https://zenodo.org/record/1194057/files/test.zip?download=1" -O "BPM2018.zip"
unzip "BPM2018.zip" -d "BPM2018"

# prep correlation tests data
echo 'Preprocessing correlation test data...'
cp preprocess.py BPM2018/correlation-tests/logs/
cd BPM2018/correlation-tests/logs
python3 preprocess.py

# prep stress tests data
echo 'Preprocessing stress test data...'
cd ../../..
cp convert-stream-to-csv.py BPM2018/stress-test/
cp preprocess-stream-csv.py BPM2018/stress-test/
cd BPM2018/stress-test/
python3 convert-stream-to-csv.py
python3 preprocess-stream-csv.py


