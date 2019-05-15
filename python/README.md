# Localization and conformance: A HMM-based approach to online conformance checking
### Authors:
- Wai Lam Jonathan Lee
- Andrea Burattin
- Jorge Munoz-Gama
- Marcos Sepulveda

## TL;DR
The python directory contains the implementation
of the proposed HMM-based approach and scripts for running experiments.

## Acknowledgement:
I want to acknowledge and thank the [hmmlearn](https://github.com/hmmlearn/hmmlearn) package; 
I had used this as a reference for my implementation of the modified HMM
proposed in the paper.

## Installation:
I recommend using the Dockerfile to do everything so that the environment setup
would be exactly the same. To run the Dockerfile, execute `make run` using the
Makefile. Once inside the container, you can install the hmmconf package by 
`python setup.py install --user`.

## Correlation test:
To run the correlation test, `cd` to `correlation-test` directory and 
just execute `python correlation_test_large.py -f
to_run` where the file `to_run` contains the names of the models that you want
to include in the test. You need to make sure that the `MODEL_DIR` and
`LOG_DIR` are correct as stated in `correlation_test_large.py` (they should be
if you are using the Docker container).

## Stress test:
Similar to the correlation tests, `cd` to `stress-test` directory and just
execute `python stress_test.py`. 

## Real-life test:
Similar to the correlation tests, `cd` to `real-life-test` directory and just
exeucte `python road-traffic.py`.

