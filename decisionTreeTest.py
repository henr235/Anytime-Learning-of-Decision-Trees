from decisionTree import DecisionTree
from decisionTreePlot import DecisionTreePlot

if __name__ == "__main__":
    trainingSetFile = open('SampleSets/training_set.csv')
    trainingSetData = trainingSetFile.readlines()
    trainingSetFile.close()
    decisionTree = DecisionTree(trainingSetData)
    #Algorithm Type - Example
    decisionTree.C45()
    #decisionTree.ID3()
    #decisionTree.ID3K(2)
    #decisionTree.SID3()
    #decisionTree.LSID3(3)
    #decisionTree.LSID3PathSample(2)
    #decisionTree.LSID3MC(1, 0.1)
    #decisionTree.BLSID3(1)
    #decisionTree.BLSID3PathSample(1)
    #decisionTree.LSID3Sequenced(2)
    #decisionTree.IIDT(10, 0.5)

    print("****Tree Data BEFORE Pruning****")  
    print("Tree Size - Number of Nodes:", decisionTree.size())
    print("Number of Leafs:", decisionTree.getNumLeafs())
    print("Tree Depth:", decisionTree.getTreeDepth())
    testSetFile = open('SampleSets/test_set.csv')
    testSetData = testSetFile.readlines()
    testSetFile.close()
    print("Prediction:", str(decisionTree.predict(testSetData)*100) + "%")
    #decisionTreePlot = DecisionTreePlot()
    #decisionTreePlot.createDecisionTreePlot(decisionTree)
    
    print("****Tree Data AFTER Pruning****")
    validationSetFile = open('SampleSets/validation_set.csv')
    validationSetData = validationSetFile.readlines()
    validationSetFile.close()
    decisionTree.prune(validationSetData)
    print("Tree Size - Number of Nodes:", decisionTree.size())
    print("Number of Leafs:", decisionTree.getNumLeafs())
    print("Tree Depth:", decisionTree.getTreeDepth())
    print("Prediction:", str(decisionTree.predict(testSetData)*100) + "%")
    #decisionTreePlot = DecisionTreePlot()
    #decisionTreePlot.createDecisionTreePlot(decisionTree)

