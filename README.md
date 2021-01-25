# Anytime Learning of Decision Trees
A python implementation of Decision Trees algorithms described in the paper ["Anytime Learning of Decision Trees"](https://www.jmlr.org/papers/volume8/esmeir07a/esmeir07a.pdf) written by Saher Esmeir and Shaul Markovitch.

## Decision Trees Algorithms
* ID3
* C4.5
* ID3-K - A lookahead version of ID3
* SID3 - Stochastic ID3
* LSID3 - Lookahead by Stochastic ID3 (Multiway splits)
* BLSID3 - Binary splits version of LSID3
* LSID3PathSample - Variation of LSID3, instead of using SID3 in order to sample trees, we use it in order to sample different tree paths (Multiway splits)
* BLSID3PathSample - Binary splits version of LSID3PathSample
* LSID3Sequenced - Conversion of LSID3 to an interruptible algorithm by sequenced invocations
* LSID3-MC - Monte Carlo evaluation of LSID3
* IIDT - Interruptible Induction of Decision Trees
* Pruning Method - Error Reduced Pruning
