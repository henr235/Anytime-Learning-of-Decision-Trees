# Anytime Learning of Decision Trees
A python implementation of Decision Tree algorithms described in the paper ["Anytime Learning of Decision Trees"](https://www.jmlr.org/papers/volume8/esmeir07a/esmeir07a.pdf) written by Saher Esmeir and Shaul Markovitch.

## Decision Tree Algorithms
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

## Prerequisites
* Python 3.6+ ([Download Page](https://www.python.org/downloads/)). You can check python version with the command: `python --version`.

## Usage
This example will demonstrate how to use the Decision Tree Algorithms:
```python
from decisionTree import DecisionTree

trainingSetFile = open('SampleSets/training_set.csv')
trainingSetData = trainingSetFile.readlines()
trainingSetFile.close()
decisionTree = DecisionTree(trainingSetData)
decisionTree.LSID3(3) # Using LSID3 with r=3
print("Tree Size - Number of Nodes:", decisionTree.size())
print("Number of Leafs:", decisionTree.getNumLeafs())
print("Tree Depth:", decisionTree.getTreeDepth())
```
Example output:
```json
Tree Size - Number of Nodes: 321
Number of Leafs: 159
Tree Depth: 9
```

For more examples and information regarding the Decision Tree Algorithms, please view the test file decisionTreeTest.py.
