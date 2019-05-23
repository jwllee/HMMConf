# Localization and conformance: A HMM-based approach to online conformance checking
### Authors:
- Wai Lam Jonathan Lee
- Andrea Burattin
- Jorge Munoz-Gama
- Marcos Sepulveda

## TL;DR
The python directory contains the implementation
of the proposed HMM-based approach and scripts for running experiments.

## Implementation of prefix alignment:
This is implemented by modifying the `isFinalMarking` function of the
`SyncProduct` class. To only require that the log projection of the alignment
corresponds to the trace, we require that the sink place of the trace net to
have at least one token. Given the A\* algorithm property, the alignment we get
at the first time we achieve the condition should be optimal.
