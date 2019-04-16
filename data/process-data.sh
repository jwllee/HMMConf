#!/usr/bin/env bash 
wget "https://zenodo.org/record/1194057/files/test.zip?download=1" -O "BPM2018.zip"
unzip "BPM2018.zip" -d "BPM2018"
cp preprocess.py BPM2018/correlation-tests/logs/
cd BPM2018/correlation-tests/logs
python preprocess.py
