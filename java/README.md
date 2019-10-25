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
have at least one token. Given the A\* algorithm property, when we achieve this
condition for the first time, the alignment we get should be optimal.

## Docker environment:
- Need to download the jdk-installer.tar.gz of Java 8 from https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html

## How to get lpsolve working in Eclipse
1. Get the lpsolve dlls from working installation of ProM or go to https://github.com/jwllee/LAC/tree/master/java
2. At Eclipse, go to Project (tab) -> Properties -> Java Build Path -> Libraries (tab) 
3. Find the lpsolve jar in Ivy and set Native library location to the folder containing the lpsolve dlls
