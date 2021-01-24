import math
import random

def getRandomValueIndexByProportionalProbability(arrayValue):
    """
    Choose a value at random from array of values,
    the probability of selecting the value is proportional
    to the other value in the array.

    Parameters
    ----------
    arrayValue : list
        list of integer values.

    Returns
    -------
    int
        A value that represents an index of the value in a list.
    """
    
    indexValue = -1
    if len(arrayValue) == 0:
        return indexValue
    copyArrayValue = arrayValue.copy()
    sumArrayValue = 0
    #Sums each value with all the values until each index (the cumulative sum of all the values until current index)
    for i, value in zip(range(len(arrayValue)), arrayValue):
        sumArrayValue += arrayValue[i]
        copyArrayValue[i] = sumArrayValue
    randomValue = random.uniform(0, sumArrayValue)
    #Find the index in proportionalto its value
    for i, value in zip(range(len(copyArrayValue)), copyArrayValue):
        #In case of an error
        if randomValue > sumArrayValue:
            break
        if value <= randomValue:
            indexValue = i
        else:
            break
    return indexValue

def rearrangeTestData(testData):
    """
    Rearrange a given data set into 3 different lists:
    attribute list, sample list and label list.

    Parameters
    ----------
    testData : list
        A list that represents a data set.

    Returns
    -------
    list
        A list that represents the attribues in the data set.
    list
        A list that represents the samples in the data set.
    list
        A list that represents the labels in the data set.
    """
    
    copyTestData = testData.copy()
    for i in range(len(copyTestData)):
        copyTestData[i] = copyTestData[i].strip().split(',')
    testAttributes = copyTestData[0].copy()
    #Remove the last value which is the lable value
    del testAttributes[-1]
    testSamples = copyTestData.copy()
    #Remove the first line which is the attributes value
    del testSamples[0]
    #Gets all the correct Labels
    testLabels = []
    for s in testSamples:
        testLabels.append(s.pop())
    return testAttributes, testSamples, testLabels
