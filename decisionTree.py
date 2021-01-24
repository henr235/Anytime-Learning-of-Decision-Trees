import math
import random
import time
from collections import Counter
from decisionTreeNode import DecisionTreeNode
import auxFunc

class DecisionTree(object):
    def __init__(self, *args, **kwargs):
        """
        The Multiple Constructor for DecisionTree class.
        
        Parameters
        ----------
        In case one Parameter was given:
        trainingSetData : list
            A list that represents all the data set information.

        In case 3 Parameters were given:
        sample : list
            A List of samples from a selected data.
        attributes : list
            A List of attributes that represents the values in the data.
        labels : list
            A List of labels (or classes) that represents the samples in the data.
        """

        if len(args) == 1:
            self.attributes, self.sample, self.labels = auxFunc.rearrangeTestData(args[0])
        else:
            self.sample = args[0]
            self.attributes = args[1]
            self.labels = args[2]
        self.labelTypes = None #A List of all the different labels (without duplicates)
        self.labelTypesCount = None #A List that represents the amount of appearances for each lable in the data set
        self.root = None #Tree Root pointer
        self.initLabelTypes() #Initiate the labelTypes and labelTypesCount values

#*************************************GENERAL TREE METHODS*************************************#
    def size(self, currentNode = None):
        """
        Calculate recursively the tree size from a particular node in the tree.
        (Gets the number of nodes in the sub tree that the current node is its root).

        Parameters
        ----------
        currentNode : Node, optional
            A Node class parameter that represents a node in the tree (default is None).
            In case currentNode is None, the parameter will represent the tree root.      

        Returns
        -------
        int
            A value that represents the number of nodes in the sub tree that currentNode is its root.
        """
        
        if currentNode is None:
            currentNode = self.root
        if currentNode.children is None:
            return 1
        listValues = []
        for child in currentNode.children:
            #Recursively going through all the nodes in the tree
            listValues.append(self.size(child.next))
        return (sum(listValues) + 1)
    

    #Calculate the number of leafs in the tree from a particular node
    def getNumLeafs(self, currentNode = None):
        """
        Calculate recursively the number of leafs from a particular node in the tree.

        Parameters
        ----------
        currentNode : Node, optional
            A Node class parameter that represents a node in the tree (default is None).
            In case currentNode is None, the parameter will represent the tree root.      

        Returns
        -------
        int
            A value that represents the number of leafs in the sub tree that currentNode is its root.
        """
        
        if currentNode is None:
            currentNode = self.root
        if currentNode.children is None:
            return 1
        listValues = []
        for child in currentNode.children:
            #Recursively going through all the nodes in the tree
            listValues.append(self.getNumLeafs(child.next))
        return (sum(listValues))

    #Calculate the max depth of the tree from the a particular node 
    def getTreeDepth(self, currentNode = None):
        """
        Calculate recursively the tree max depth from a particular node in the tree.

        Parameters
        ----------
        currentNode : Node, optional
            A Node class parameter that represents a node in the tree (default is None).
            In case currentNode is None, the parameter will represent the tree root.      

        Returns
        -------
        int
            A value that represents the max depth in the sub tree that currentNode is its root.
        """
        
        if currentNode is None:
            currentNode = self.root
        if currentNode.children is None:
            return 1
        listValues = []
        for child in currentNode.children:
            #Recursively going through all the nodes in the tree
            listValues.append(self.getTreeDepth(child.next))
        return (max(listValues) + 1)

    def getInnerNodesInSubTree(self, currentNode, listNode):
        """
        Gets recursively all the inner nodes from a particular node in the tree.

        Parameters
        ----------
        currentNode : Node
            A Node class parameter that represents a node in the tree.
        listNode : list
            An empty list parameter that represents a list of nodes from the tree.
            When the recursion is done, the listNode will contain all the inner nodes
            from the sub tree if there are any.
        """
        
        if currentNode.children:
            listNode.append(currentNode)
            if currentNode.children:
                for child in currentNode.children:
                    #Recursively going through all the nodes in the tree
                    self.getInnerNodesInSubTree(child.next, listNode)
                    
    def getLeafNodesInSubTree(self, currentNode, listLeaf):
        """
        Gets recursively all the leaf nodes from a particular node in the tree.

        Parameters
        ----------
        currentNode : Node
            A Node class parameter that represents a node in the tree.
        listLeaf : list
            An empty list parameter that represents a list of nodes from the tree.
            When the recursion is done, the listLeaf will contain all the leaf nodes
            from the sub tree if there are any.
        """
        
        if currentNode.children is None:
            listLeaf.append(currentNode)
        else:
            for child in currentNode.children:
                #Recursively going through all the nodes in the tree
                self.getLeafNodesInSubTree(child.next, listLeaf)

    def getNodeAncestors(self, currentNode):
        """
        Gets all the node ancestors of a particular node in the tree.

        Parameters
        ----------
        currentNode : Node
            A Node class parameter that represents a node in the tree.     

        Returns
        -------
        list
            A list of node ancestors of a particular node in the tree.
        """
        
        nodeAncestors = []
        if currentNode is None or currentNode.parent is None:
            return nodeAncestors
        ancNode = currentNode
        #Going through the node parents until we get to the tree root
        while ancNode.parent:
            ancNode = ancNode.parent.parent
            nodeAncestors.append(ancNode)
        return nodeAncestors

    def getRandomPathLength(self):
        """
        Calculate the length of a random path from the tree root to one of its leafs.    

        Returns
        -------
        int
            A value that represents the length of a random from the tree root to one of its leafs.
        """
        
        pathSize = 1
        currentNode = self.root
        #Go through the tree and find a path at random
        while currentNode.children:
            copyChildren = (currentNode.children).copy()
            #Shuffling the children list in order to get real coincidental
            random.shuffle(copyChildren)
            childSelected = copyChildren[0]
            currentNode = childSelected.next
            pathSize += 1
        return pathSize
#***********************************END GENERAL TREE METHODS***********************************#

  
#********************************GENERAL DECISION TREE METHODS*********************************#
    def initLabelTypes(self):
        """
        Initiate the labelTypes and labelTypesCount values that are defined in the Constructor.    
        """
        
        self.labelTypes = []
        self.labelTypesCount = []
        for lab in self.labels:
            #In case we come across a new type of label
            if lab not in self.labelTypes:
                self.labelTypes.append(lab)
                self.labelTypesCount.append(0)
            #In case the label already exists we update the amount of appearance
            self.labelTypesCount[self.labelTypes.index(lab)] += 1

    def getLabelTypeId(self, sampleId):
        """
        Gets the index value in a list that represents a label type of a give sample.

        Parameters
        ----------
        sampleId : int
            An index that represents a sample from the sample data.

        Returns
        -------
        int
            An index value in a list that represents a label type of a give sample.
        """
        
        return self.labelTypes.index(self.labels[sampleId])

    def getDominantLabel(self, sampleIds):
        """
        Gets the label that appears the most for a list of samples.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.     

        Returns
        -------
        any
            A value that represents a label type of a give list of samples.
        """
        
        labelTypesCount = [0] * len(self.labelTypes)
        #Go through all the samples and count the amount of different labels that we come across
        for sid in sampleIds:
            labelTypesCount[self.labelTypes.index(self.labels[sid])] += 1
        return self.labelTypes[labelTypesCount.index(max(labelTypesCount))]

    def isSingleLabeled(self, sampleIds):
        """
        Check if a list of samples is represented only by a single label.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.     

        Returns
        -------
        bool
            Return True if the samples represented by a single label, False otherwise.
        """
        if len(sampleIds) == 0:
            return False
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False
        return True

    def getAttributeValues(self, sampleIds, attributeId):
        """
        Calculate a list of values that represents a given attribute and data sample.
        In other words, calculate the domain of a given attribute for a given data.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeId : int
            An index that represents an attribute from the attribute list.  

        Returns
        -------
        list
            A list of values that represents the attribute (the attributes Domain).
        """
        
        attributeValues = []
        for sid in sampleIds:
            value = self.sample[sid][attributeId]
            #Append only different values to the list
            if value not in attributeValues:
                attributeValues.append(value)
        return attributeValues

    def getMinDomainAttFromListAtt(self, sampleIds, attributeIds):
        """
        Calculate the minimum domain size among all the given attributes. 

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.  

        Returns
        -------
        int
            A value of the minimum domain size among all the given attributes.
        """
        
        minDomainAttID = -1
        minDomainSize = math.inf
        #Loop through each attribute and find the one with min domain size
        for attId in attributeIds:
            currentDomainSize = len(self.getAttributeValues(sampleIds, attId))
            #Check for new min value
            if currentDomainSize < minDomainSize:
                minDomainSize = currentDomainSize
                minDomainAttID = attId
        return minDomainSize
#******************************END GENERAL DECISION TREE METHODS*******************************#


#******************************DECISION TREE INFORMATION METHODS*******************************#
    def calculateEntropy(self, sampleIds):
        """
        Calculate the Entropy of a given sample data.
        Entropy is a measure of the amount of uncertainty in the (data) set.
        This method mainly used to calculate decision tree using ID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.

        Returns
        -------
        double
            A value of the calculated entropy of a given sample data.
        """
        
        entropy = 0
        labelCount = [0] * len(self.labelTypes)
        #Calculate the amount of each label (class) from a given samples
        for sid in sampleIds:
            labelCount[self.getLabelTypeId(sid)] += 1
        #For each label we need to update to the correct entropy
        for labelAmount in labelCount:
            if labelAmount > 0:
                pValue = (labelAmount/len(sampleIds))
                entropy -= (pValue * math.log(pValue, 2))
        return entropy

    def calculateInformationGain(self, sampleIds, attributeId, algoType = 0):
        """
        Calculate the Information Gain with a given sample data,
        attribute and algorithm type.
        Information Gain is the measure of the difference in entropy from before
        and after the set is split on an attribute.
        In other words, how much uncertainty was reduced after splitting set on an attribute.
        This method mainly used to calculate decision tree using C4.5 and ID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeId : int
            An index that represents an attribute from the attribute list.  
        algoType : int, optional
            This value will decide the algorithm type to use:
            when algoType = 0 - The calculation will be for ID3 algorithm.
            when algoType = 1 - The calculation will be for C4.5 algorithm.
            (default value is 0)

        Returns
        -------
        double
            A value of the calculated Information Gain of a given sample data,
            attribute and algorithm type.
        """
        
        gain = self.calculateEntropy(sampleIds)
        iv = 0
        attributeValues = []
        attributeValuesCount = []
        attributeValuesIds = []
        #Calculate domain values, value amount and value sample ids
        for sid in sampleIds:
            value = self.sample[sid][attributeId]
            if value not in attributeValues:
                attributeValues.append(value)
                attributeValuesCount.append(0)
                attributeValuesIds.append([])
            valueId = attributeValues.index(value)
            attributeValuesCount[valueId] += 1
            attributeValuesIds[valueId].append(sid)
        #For each value in the domain we need to update to the correct gain
        for valueCount, valueIds in zip(attributeValuesCount, attributeValuesIds):
            pValue = (valueCount/len(sampleIds))
            gain -= (pValue * self.calculateEntropy(valueIds))
            iv -= (pValue * math.log(pValue, 2)) #Used for C4.5
        #In case we are using C4.5 algorithm
        if algoType == 1:
            if iv == 0:
                return -math.inf
            #Normalization of the gain value
            return (gain/iv)
        #In case we are using ID3 algorithm we simply return the calculated gain
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds, algoType = 0):
        """
        Calculate the maximum Information Gain with a given sample data,
        between a list of attributes and algorithm type.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.  
        algoType : int, optional
            This value will decide the algorithm type to use:
            when algoType = 0 - The calculation will be for ID3 algorithm.
            when algoType = 1 - The calculation will be for C4.5 algorithm.
            (default value is 0)

        Returns
        -------
        any
            A value that represents an attribute with maximum informatio gain.
        int
            An index that represents an attribute with maximum informatio gain. 
        """
        
        attributesEntropy = [0] * len(attributeIds)
        #Calculating each atrributes information gain
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.calculateInformationGain(sampleIds, attId, algoType)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        return self.attributes[maxId], maxId
#****************************END DECISION TREE INFORMATION METHODS*****************************#


#***********************************ID3K INFORMATION METHODS***********************************#
    def calculateEntropyK(self, sampleIds, attributeIds, attributeId, kValue):
        """
        Calculate the Entropyfor lookahead-based variation of ID3.
        Entropy is a measure of the amount of uncertainty in the (data) set.
        This method mainly used to calculate decision tree using ID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data. 
        attributeId : int
            An index that represents an attribute from the attribute list.
        kValue : int
            A value that represents the lookahead amount.

        Returns
        -------
        double
            A value of the calculated entropy of a given sample data,
            a list of attributes, attribute and lookahead value.
        """
        
        #In case the value of k is not leagal or there no attributes but k is still greater than 0
        if (kValue <= 0 or len(attributeIds) == 0):
            return self.calculateEntropy(sampleIds)
        attributeIdsCopy = attributeIds.copy()
        sumReturn = 0
        #We loop through the domain of attribute attributeId
        for value in self.getAttributeValues(sampleIds, attributeId):
            #eSampleIds is the list that represents the Ei={e in E|a(e)=vi}, a is attribute and vi is a value in domain(a) 
            eSampleIds = sampleIds.copy()
            for sid in sampleIds:
                if self.sample[sid][attributeId] != value:
                    eSampleIds.remove(sid)
            eSampleCount = len(eSampleIds)
            #end of eSampleIds calculation
            #We loop through all the attributes to calculate each entropy
            minekValue = math.inf
            for attId in attributeIds:
                attributeIdsCopy.remove(attId)
                #Calculate the minimum value
                ekValue = self.calculateEntropyK(eSampleIds, attributeIdsCopy, attId, kValue - 1)
                if minekValue > ekValue:
                    minekValue = ekValue
                attributeIdsCopy.append(attId)
            sumReturn += ((eSampleCount/len(sampleIds)) * minekValue)
        return sumReturn

    def calculateInformationGainK(self, sampleIds, attributeIds, attributeId, kValue):
        """
        Calculate the Information Gain for lookahead-based variation of ID3.
        Information Gain is the measure of the difference in entropy from before
        and after the set is split on an attribute.
        In other words, how much uncertainty was reduced after splitting set on an attribute.
        This method mainly used to calculate decision tree using ID3K algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data. 
        attributeId : int
            An index that represents an attribute from the attribute list.
        kValue : int
            A value that represents the lookahead amount.

        Returns
        -------
        double
            A value of the calculated Information Gain of a given sample data,
            a list of attributes, attribute and lookahead value.
        """
                
        return self.calculateEntropy(sampleIds) - self.calculateEntropyK(sampleIds, attributeIds, attributeId, kValue)

#*********************************END ID3K INFORMATION METHODS*********************************#


#************************************PREDICT DECISION TREE*************************************#
    #Use the Test data to predict tree accuracy
    def predict(self, testData):
        """
        Calculate the accuracy of the trained decision tree on the test set.

        Parameters
        ----------
        testData : list
            A list that represents the test set.

        Returns
        -------
        float
            A list that represents the accuracy of the trained decision tree on the test set.
        """
        
        testAttributes, testSamples, testLabels = auxFunc.rearrangeTestData(testData)
        accuracyAmount = 0
        for i, testSample in zip(range(len(testSamples)), testSamples):
            currentNode = self.root
            #Go through the tree and find the prediction according to testSample
            while currentNode.children:
                #The value here is the current attribute
                attNode = currentNode.value
                #Get the index of the attribute in the data test
                attTestAllIndex = [i for i, s in enumerate(testAttributes) if attNode == s]
                if len(attTestAllIndex) == 0:
                    break
                attTestIndex = attTestAllIndex[0]
                foundChild = 0
                #Go through all the node childern to find correct value
                for child in currentNode.children:
                    if child.value == testSample[attTestIndex]:
                        currentNode = child.next
                        foundChild = 1
                        break #if found, no need to continue
                #In case a child was not found we need the search for a child with a start value of '!'
                if foundChild == 0:
                    for child in currentNode.children:
                        if (child.value)[0] == '!':
                            currentNode = child.next
                            foundChild = 1
                            break
                #If still not found, the value does not exist in the tree, then we can finish
                if foundChild == 0:
                    break
            #Evaluate prediction when we reach a leaf
            if currentNode.value == testLabels[i]:
                accuracyAmount += 1
        #Calculate accuracy
        accuracyCal = (accuracyAmount / len(testLabels))
        return accuracyCal
#**********************************END PREDICT DECISION TREE***********************************#


#************************************REDUCED ERROR PRUNING*************************************#
    def prune(self, validationSet):
        """
        Pruning a tree using Reduced Error Pruning technique.
        After using this method, the outcome will be a pruned tree if the tree
        can be pruned.

        Parameters
        ----------
        validationSet : list
            A list that represents the validation set.
        """
        
        #First we need to calculate the majority class (lable) and error for all the nodes using a validation set
        self.calculateMajorityClassAndErrorForPruning(validationSet)
        flag = True
        #Check all the variations of leafs in the tree until we can not prune anymore (meaning, the error value all parents is bigger then all their children)
        while flag:
            #Go through all the nodes in bottom-up fashion
            nodesWithOnlyLeafChildren = self.getNodesWithOnlyLeafChildren()
            #In case of an error
            if len(nodesWithOnlyLeafChildren) == 0:
                break
            for currentNode in nodesWithOnlyLeafChildren:
                stopCriteria = True
                canPrune = False
                #Check for pruning only if there is error in the node that is not None
                if currentNode.errorValue is not None:
                    #Check each child leaf of the node:
                    #If the error value of one of the children is bigger then the parent, we can turn the paren into a leaf and its value will be the majority Class
                    for childLeaf in currentNode.children:
                        if childLeaf.next.errorValue is not None and currentNode.errorValue < childLeaf.next.errorValue:
                            #If we are here, we can prune
                            canPrune = True
                            break #we can stop the loop if Criteria for Reduced Error Pruning is met
                    if canPrune:
                        #Actual Pruning
                        currentNode.children = None
                        currentNode.value = currentNode.majorityClass
                        canPrune = False
                        stopCriteria = False
            if stopCriteria:
                flag = False    
                
    #Calculate the majority class (lable) and error for all the nodes using a validation set
    def calculateMajorityClassAndErrorForPruning(self, validationSet):
        """
        Calculate the majority class (lable) and error for all the nodes using a validation set.
        After using this method, the result will be updated in the majorityClass value and
        the errorValue for each node.

        Parameters
        ----------
        validationSet : list
            A list that represents the validation set.
        """
        
        validationAttributes, validationSamples, validationLabels = auxFunc.rearrangeTestData(validationSet)
        #Calculate the majority class (lable) and error for all the nodes that are not leafs
        listNode = []
        self.getInnerNodesInSubTree(self.root, listNode)
        for currentNode in listNode:
            #Get all the values in the path from root to the current node in the tree
            newPathValues = self.getPathValuesFromRootToNode(currentNode)
            #Get all the labels that we come across when we take the path from the root to the current node using the validation set
            nodeLabels = self.getNodeLabels(newPathValues, validationAttributes, validationSamples, validationLabels)
            if len(nodeLabels) != 0:
                #In case there are no lables (len(nodeLabels) == 0), pruning will not improve accuracy, so we will not prune this node
                #Calculate the number of occurrences for the different label values using Counter function (the new value will be a dictionary)
                labelOccurrences = Counter(nodeLabels)
                #We choose the majority class (lable) to represent each node as error value
                maxLabelOccurrence = max(labelOccurrences.values())  # max value
                maxLabelsList = [k for k, v in labelOccurrences.items() if v == maxLabelOccurrence] # getting all keys containing the the max value
                maxLabel = maxLabelsList[0]
                errorValue = (len(nodeLabels) - maxLabelOccurrence)
                currentNode.majorityClass = maxLabel
                currentNode.errorValue = errorValue
        #Calculate the majority class (lable) and error for all leaf nodes
        listLeaf = []
        self.getLeafNodesInSubTree(self.root, listLeaf)
        for currentNode in listLeaf:
            #Get all the values in the path from root to the current node in the tree
            newPathValues = self.getPathValuesFromRootToNode(currentNode)
            #Get all the labels that we come across when we take the path from the root to the current node using the validation set
            nodeLabels = self.getNodeLabels(newPathValues, validationAttributes, validationSamples, validationLabels)
            leafValueLabel = currentNode.value
            if len(nodeLabels) != 0:
                #Calculate the number of occurrences for the different label values using Counter function (the new value will be a dictionary)
                labelOccurrences = Counter(nodeLabels)
                #Check if the leaf value label exists in order to calculate the error amount
                leafValueOccurrences = labelOccurrences.get(leafValueLabel)
                if leafValueOccurrences is None:
                    errorValue = len(nodeLabels)
                else:
                    errorValue = (len(nodeLabels) - leafValueOccurrences)
                currentNode.errorValue = errorValue
        #In any other calculation possibility for the majority class (lable) and error, these values will remain None
        
    def getPathValuesFromRootToNode(self, currentNode):
        """
        Gets all the attributes and their values in the path created from
        the tree root to the current node.

        Parameters
        ----------
        currentNode : Node
            A Node class parameter that represents a node in the tree.     

        Returns
        -------
        dictionary
            A dictionary that represents path values with attribute as key and attribute value as value.
        """
        
        pathValues = {}
        nodePtr = currentNode
        #Go through all the parents to create the path that leads to the root from the current node
        while nodePtr.parent:
            if nodePtr == self.root:
                break
            pathValues[nodePtr.parent.parent.value] = nodePtr.parent.value
            nodePtr = nodePtr.parent.parent
        return pathValues

    def getNodeLabels(self, nodePathValues, validationAttributes, validationSamples, validationLabels):
        """
        Calculate all the labels (classes) that we come across in the path from
        the root to a certain node using the validation set.

        Parameters
        ----------
        nodePathValues : dictionary
            A dictionary that represents path values with attribute as key and attribute value as value.     

        Returns
        -------
        list
            A list that contains all the labels in the path.
        """
        
        copyValidLabels = validationLabels.copy()
        #Go through all Attributes from root to the the node using nodePathValues and gets all the sample ids that intersects between all the values of the attributes
        for attName, attValue in nodePathValues.items():
            attSampleIds = []
            #Get the index of the attribute in the data test
            attValidAllIndex = [i for i, s in enumerate(validationAttributes) if attName == s]
            if len(attValidAllIndex) == 0:
                break
            attValidIndex = attValidAllIndex[0]
            #Get Only samples with attribute value
            for iValue, sIdValue in zip(range(len(validationSamples)), validationSamples):
                if validationSamples[iValue][attValidIndex] != attValue:
                    copyValidLabels[iValue] = None
        #Get only needed Lables (Lables that are not None)
        neededLabels = [e for e in copyValidLabels if e is not None]
        return neededLabels

    #
    def getNodesWithOnlyLeafChildren(self):
        """
        Gets all nodes from the tree root that all their childern are leafs.   

        Returns
        -------
        list
            A list of nodes that all their childern are leafs.
        """
        
        #Get all the nodes that are not leafs
        listNode = []
        self.getInnerNodesInSubTree(self.root, listNode)
        nodesWithOnlyLeafChildren = []
        #Go through all the nodes and check which when as only leafs as children
        for currentNode in listNode:
            #Note that currentNode children can not be None, because the listNode contain inly nodes that has children
            leafCount = 0
            for child in currentNode.children:
                if child.next.children is None:
                    leafCount += 1
            #Check if the number of childern equal to leaf count
            if leafCount == len(currentNode.children):
                #Only if equal we append the node
                nodesWithOnlyLeafChildren.append(currentNode)
        return nodesWithOnlyLeafChildren
#**********************************END REDUCED ERROR PRUNING***********************************#


#*****************************CHOOSING ATTRIBUTE FOR DECISION TREE*****************************#
    def chooseAttributeC45(self, sampleIds, attributeIds):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for C4.5 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.  

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
        
        return self.getAttributeMaxInformationGain(sampleIds, attributeIds, 1)
    
    def chooseAttributeID3(self, sampleIds, attributeIds):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for ID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.  

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
                
        return self.getAttributeMaxInformationGain(sampleIds, attributeIds)

    def chooseAttributeID3K(self, sampleIds, attributeIds, kValue):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for ID3K algorithm.
        The best attribute is the one with the maximum Information Gain with
        a k value.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        kValue : int
            A value that represents the lookahead amount.

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
        
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.calculateInformationGainK(sampleIds, attributeIds, attId, kValue)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        return self.attributes[maxId], maxId

    def chooseAttributeSID3(self, sampleIds, attributeIds):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for SID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.  

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
        
        attInformationGain = attributeIds.copy()
        returnId = attributeIds[0]
        #Calculate the Information Gain for each attribute using calculateInformationGainK for k=1
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attInformationGain[i] = self.calculateInformationGainK(sampleIds, attributeIds, attId, 1)
        #Calculate the entropy for each attribute using calculateEntropyK for k=1
        #in order to see if exists entropy that is zero
        attributesIdEntropyZero = []
        for attId in attributeIds:
            if self.calculateEntropyK(sampleIds, attributeIds, attId, 1) == 0:
                attributesIdEntropyZero.append(attId)
        #If there is an Attribute with entropy zero then we choose randomly from attributesIdEntropyZero
        if (len(attributesIdEntropyZero) != 0):
            random.shuffle(attributesIdEntropyZero)
            returnId = attributesIdEntropyZero[0]
        else:
            #Choose attribute at random from the attribute list; for each attribute,
            #the probability of selecting it is proportional to its Information Gain with k=1
            indexValue = auxFunc.getRandomValueIndexByProportionalProbability(attInformationGain)
            #In case of an error
            if indexValue != -1:
                returnId = attributeIds[indexValue]
        return self.attributes[returnId], returnId

    def chooseAttributeLSID3(self, sampleIds, attributeIds, rValue, samplingType = 0):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for LSID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        rValue : int
            A value that represents the lookahead amount.

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
        
        if rValue == 0:
            return self.chooseAttributeID3(sampleIds, attributeIds)
        attributesSumSID3 = [0] * len(attributeIds)
        attributeIdsCopy = attributeIds.copy()
        #Going through every attribute in the attribute list
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            #We loop through the domain for each attribute attId
            for value in self.getAttributeValues(sampleIds, attId):
                #eSample is the list that represents the Ei={e in E|a(e)=vi}, a is attribute and vi in domain(a) 
                eSample = self.sample.copy()
                newLabels = self.labels.copy()
                #Calculate only needed values
                for sid in sampleIds:
                    if self.sample[sid][attId] != value:
                        eSample[sid] = None
                        newLabels[sid] = None
                #Remove samples that are not in the sampleIds list
                for idSample in range(len(self.sample)):
                    if idSample not in sampleIds:
                        eSample[idSample] = None
                        newLabels[idSample] = None
                #Get only necessary values
                eSample = [e for e in eSample if e is not None]
                newLabels = [e for e in newLabels if e is not None]
                #end of eSample calculation
                #Create new list of attributes using real value
                attributeIdsCopy.remove(attId)
                currentAttributes = []
                for attributeIdInList in attributeIdsCopy:
                    currentAttributes.append(self.attributes[attributeIdInList])
                minSID3 = math.inf
                looprValue = rValue
                #Going over all the r values
                while looprValue > 0:
                    looprValue -= 1
                    decisionTree = DecisionTree(eSample, currentAttributes, newLabels)
                    decisionTree.SID3()
                    #Sampling the value through a random FULL tree size (using SID3)
                    if samplingType == 0:
                        valueSID3 = decisionTree.size()
                    #Sampling the value through a random path sample from the tree (using getRandomPathLength form SID3)
                    else:
                        valueSID3 = decisionTree.getRandomPathLength()
                    #Checking for the minimum value
                    if minSID3 > valueSID3:
                        minSID3 = valueSID3
                attributeIdsCopy.append(attId)
                #Sum all min values of tree or path sampling for the values in the attribute domain
                attributesSumSID3[i] += minSID3
        minAttId = attributeIds[attributesSumSID3.index(min(attributesSumSID3))]
        return self.attributes[minAttId], minAttId

    def chooseAttributeLSID3MC(self, sampleIds, attributeIds, rValue, pValue):
        """
        Calculate the best attribute with of a given sample data,
        between a list of attributes for LSID3MC algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        rValue : int
            A value that represents the lookahead amount.
        pValue : double
            A value that represents sample size.

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """
        
        if rValue == 0:
            return self.chooseAttributeID3(sampleIds, attributeIds)
        attributesSumValues = [0] * len(attributeIds)
        #Going through every attribute in the attribute list
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            minValue = math.inf
            looprValue = rValue
            #Going over all the r values
            while looprValue > 0:
                looprValue -= 1
                valueMC, bestTest = self.evaluateContinuousAttributeMC(sampleIds, attributeIds, attId, pValue)
                #Checking for the minimum value
                if minValue > valueMC:
                    minValue = valueMC
            #Sum all min values of tree or path sampling for the values in the attribute domain
            attributesSumValues[i] += minValue
        minAttId = attributeIds[attributesSumValues.index(min(attributesSumValues))]
        return self.attributes[minAttId], minAttId

    def chooseAttributeWrapper(self, algoType, sampleIds, attributeIds, firstValue = 0, secondValue = 0):
        """
        A wrapper method for all the chooseAttribute algorithms.

        Parameters
        ----------
        algoType : str
            A value that represents the algorithm type chosen.
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        firstValue : int, optional
            A value that represents the the first value if need for the chosen algorithm.
            (default value is 0)
        secondValue : double, optional
            A value that represents the the second value if need for the chosen algorithm.
            (default value is 0)

        Returns
        -------
        any
            A value that represents the best attribute.
        int
            An index that represents the best attribute. 
        """

        if algoType == "C45":
            return self.chooseAttributeC45(sampleIds, attributeIds)
        elif algoType == "ID3K":
            return self.chooseAttributeID3K(sampleIds, attributeIds, firstValue)
        elif algoType == "SID3":
            return self.chooseAttributeSID3(sampleIds, attributeIds)
        elif algoType == "LSID3":
            return self.chooseAttributeLSID3(sampleIds, attributeIds, firstValue)
        elif algoType == "LSID3PathSample":
            return self.chooseAttributeLSID3(sampleIds, attributeIds, firstValue, 1)
        elif algoType == "LSID3MC":
            return self.chooseAttributeLSID3MC(sampleIds, attributeIds, firstValue, secondValue)
        else: #Any other case will be ID3 algorithm
            return self.chooseAttributeID3(sampleIds, attributeIds)
#***************************END CHOOSING ATTRIBUTE FOR DECISION TREE***************************#


#*******************************MC EVALUATE CONTINUOUS ATTRIBUTE*******************************#
    def evaluateContinuousAttributeMC(self, sampleIds, attributeIds, attId, pValue):
        """
        Calculate the Monte Carlo evaluation of continuous attributes using SID3.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        attId : int
            An index that represents an attribute from the attribute list.
        pValue : double
            A value that represents sample size.

        Returns
        -------
        int
            The minimum value calculated for an attribute.
        any
            A value that represents the best sample for an attribute. 
        """
        sampleIdGainCal = sampleIds.copy()
        #Going through every sample in the sample list
        for i, sid in zip(range(len(sampleIds)), sampleIds):
            #sampleEGreater is the list that represents the E_<= ={e'|a(e')<=a(e)}, a is attId, e is current sid
            sampleEGreater = sampleIds.copy()
            for sidETag in sampleIds:
                if self.sample[sidETag][attId] > self.sample[sid][attId]:
                    sampleEGreater.remove(sidETag)
            #end of sampleEGreater calculation
            #sampleESmaller is the list that represents the E_> ={e'|a(e')>a(e)}, a is attId, e is current sid
            sampleESmaller = sampleIds.copy()
            for sidETag in sampleIds:
                if self.sample[sidETag][attId] <= self.sample[sid][attId]:
                    sampleESmaller.remove(sidETag)
            #end of sampleESmaller calculation
            #Calculate sample id Entropy
            entropyEGreater = (len(sampleEGreater)/len(sampleIds)) * (self.calculateEntropy(sampleEGreater))
            entropyESmaller = (len(sampleESmaller)/len(sampleIds)) * (self.calculateEntropy(sampleESmaller))
            entropyETotal = entropyEGreater + entropyESmaller
            #Calculate sample id Gain
            sampleIdGainCal[i] = self.calculateEntropy(sampleIds) - entropyETotal
        sampleSize = (pValue * len(sampleIds))
        minValue = math.inf
        bestTest = math.inf
        sidESelected = sampleIds[0]
        loopSampleSize = sampleSize
        attributeIdsCopy = attributeIds.copy()
        #Create new list of attributes using real value
        currentAttributes = []
        for attributeIdInList in attributeIdsCopy:
            currentAttributes.append(self.attributes[attributeIdInList])
        #Going over the sample size
        while loopSampleSize > 0:
            loopSampleSize -= 1
            #Choose a sample at random from a list of samples; for each sample e,
            #the probability of selecting it is proportional to gain of the sample
            indexValue = auxFunc.getRandomValueIndexByProportionalProbability(sampleIdGainCal)
            #In case of an error
            if indexValue != -1:
                sidESelected = sampleIds[indexValue]
            #sampleEGreater is the list that represents the E_<= ={e'|a(e')<=a(e)}, a is attId, e is current sidESelected
            sampleEGreater = self.sample.copy()
            labelEGreater = self.labels.copy()
            #Calculate only needed values
            for sidETag in sampleIds:
                if self.sample[sidETag][attId] > self.sample[sidESelected][attId]:
                    sampleEGreater[sid] = None
                    labelEGreater[sid] = None
            #Remove samples that are not in the sampleIds list
            for idSample in range(len(self.sample)):
                if idSample not in sampleIds:
                    sampleEGreater[idSample] = None
                    labelEGreater[idSample] = None
            #Get only necessary values
            sampleEGreater = [e for e in sampleEGreater if e is not None]
            labelEGreater = [e for e in labelEGreater if e is not None]
            #end of sampleEGreater calculation
            #sampleESmaller is the list that represents the E_> ={e'|a(e')>a(e)}, a is attId, e is current sidESelected
            sampleESmaller = self.sample.copy()
            labelESmaller = self.labels.copy()
            #Calculate only needed values
            for sidETag in sampleIds:
                if self.sample[sidETag][attId] <= self.sample[sidESelected][attId]:
                    sampleESmaller[sid] = None
                    labelESmaller[sid] = None
            #Remove samples that are not in the sampleIds list
            for idSample in range(len(self.sample)):
                if idSample not in sampleIds:
                    sampleESmaller[idSample] = None
                    labelESmaller[idSample] = None
            #Get only necessary values
            sampleESmaller = [e for e in sampleESmaller if e is not None]
            labelESmaller = [e for e in labelESmaller if e is not None]
            #end of sampleESmaller calculation
            #Create SID3 Trees using sampleEGreater and sampleESmaller
            decisionTreeEGreater = DecisionTree(sampleEGreater, currentAttributes, labelEGreater)
            decisionTreeEGreater.SID3()
            sizeSID3EGreater = decisionTreeEGreater.size()
            decisionTreeESmaller = DecisionTree(sampleESmaller, currentAttributes, labelESmaller)
            decisionTreeESmaller.SID3()
            sizeSID3ESmaller = decisionTreeESmaller.size()
            #Calculate the value of the sample
            totalE = sizeSID3EGreater + sizeSID3ESmaller
            #Checking for the minimum value
            if totalE < minValue:
                minValue = totalE
                bestTest = self.sample[sidESelected][attId]
        return minValue, bestTest
#*****************************END MC EVALUATE CONTINUOUS ATTRIBUTE*****************************#


#************************************DECISION TREE BUILDER*************************************#
    def treeBuilder(self, sampleIds, attributeIds, root, parent, algoType, splitType = 0, firstValue = 0, secondValue = 0):
        """
        Build the decision tree recursively using the given algorithm type.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        root : Node
            A pointer to the current tree root.
        parent : Node
            A pointer to the current tree root parent.
        algoType : str
            A value that represents the algorithm type chosen.
        splitType : int, optional
            A value that represents the split type of the chosen algorithm.
            When value is 0 we use Multiway splits and for 1 we use Binary splits.
            (default value is 0)
        firstValue : int, optional
            A value that represents the first value if need for the chosen algorithm.
            (default value is 0)
        secondValue : double, optional
            A value that represents the second value if need for the chosen algorithm.
            (default value is 0)

        Returns
        -------
        Node
            A pointer to the tree root. 
        """
        
        copyAttributeIds = attributeIds.copy()
        root = DecisionTreeNode() #Initialize current root
        root.parent = parent
        #In case there is only one label
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]]
            return root
        #In case the algorithm finished going through all the attributes
        if len(copyAttributeIds) == 0:
            root.value = self.getDominantLabel(sampleIds)
            return root
        #Calculate best attribute according to the given algorithm
        bestAttrName, bestAttrId = self.chooseAttributeWrapper(algoType, sampleIds, attributeIds, firstValue, secondValue)
        root.value = bestAttrName
        root.children = []  # Create list of children
        #Check split criteria
        if splitType == 1: #Use Binary splits
            #Finds the best information gain split value from the bestAttrId domain
            bestGain = -1
            bestSid = -1
            for atriSid in sampleIds:
                atriSidAsList = [atriSid]
                infoGainAtriSid = self.calculateInformationGain(atriSidAsList, bestAttrId)
                if (infoGainAtriSid > bestGain):
                    bestGain = infoGainAtriSid
                    bestSid = atriSid
            value = self.sample[bestSid][bestAttrId]
            #Create two branches, the first will be for the best information gain value and the second if for other values
            for i in range(2):
                child = DecisionTreeNode()            
                childrenSampleIds = []
                if i == 0: #Best information gain split value
                    for sid in sampleIds:
                        if self.sample[sid][bestAttrId] == value:
                            childrenSampleIds.append(sid)
                    child.value = value
                    child.parent = root
                else: #Other domain values
                    for sid in sampleIds:
                        if self.sample[sid][bestAttrId] != value:
                            childrenSampleIds.append(sid)
                    child.value = "!" + value
                    child.parent = root
                root.children.append(child) #Append new child node to current root
                #Safe check for removing the attribute from the list
                if len(copyAttributeIds) > 0 and bestAttrId in copyAttributeIds:
                    toRemove = copyAttributeIds.index(bestAttrId)
                    copyAttributeIds.pop(toRemove)
                #Recursively compute the sub tree with the child node as a root
                child.next = self.treeBuilder(childrenSampleIds, copyAttributeIds, child.next, child, algoType, splitType, firstValue, secondValue)
        else: #Multiway splits
            #Go through the domain of the attribute in order to calculate split
            for value in self.getAttributeValues(sampleIds, bestAttrId):
                child = DecisionTreeNode() #Initialize childe Node
                child.value = value
                child.parent = root
                root.children.append(child) #Append new child node to current root
                childrenSampleIds = []
                #Calculate child samples
                for sid in sampleIds:
                    if self.sample[sid][bestAttrId] == value:
                        childrenSampleIds.append(sid)
                #Safe check for removing the attribute from the list
                if len(copyAttributeIds) > 0 and bestAttrId in copyAttributeIds:
                    toRemove = copyAttributeIds.index(bestAttrId)
                    copyAttributeIds.pop(toRemove)
                #Recursively compute the sub tree with the child node as a root
                child.next = self.treeBuilder(childrenSampleIds, copyAttributeIds, child.next, child, algoType, splitType, firstValue, secondValue)
        return root
#**********************************END DECISION TREE BUILDER***********************************#


#*******************************DECISION TREE ALGORITHM WRAPPER********************************#
    def C45(self):
        """
        A Decision Tree wrapper used for C4.5 algorithm.
        C4.5 builds decision trees from a set of training data in the same way as ID3,
        using the concept of information entropy.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "C45")

    def ID3(self):
        """
        A Decision Tree wrapper used for ID3 algorithm.
        ID3 (Iterative Dichotomiser 3) builds decision trees from a set of training data,
        it iterates through every unused attribute of the set and calculates the entropy
        or the information gain of that attribute, then select the one with largest information gain.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "ID3")

    def ID3K(self, kValue):
        """
        A Decision Tree wrapper used for ID3K algorithm.
        ID3K is a lookahead based variation of ID3 algorithm.

        Parameters
        ----------
        kValue : int
            A value that represents the lookahead amount.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "ID3K", kValue)

    def SID3(self):
        """
        A Decision Tree wrapper used for SID3 algorithm.
        SID3 is a Stochastic variation of ID3 algorithm.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "SID3")

    def LSID3(self, rValue):
        """
        A Decision Tree wrapper used for LSID3 algorithm.
        LSID3 is a Lookahead Stochastic variation of ID3 algorithm.
        This algorithm is estimating tree size by sampling trees using SID3 algorithm.

        Parameters
        ----------
        rValue : int
            A value that represents the lookahead amount.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "LSID3", 0, rValue)

    def LSID3PathSample(self, rValue):
        """
        A Decision Tree wrapper used for LSID3PathSample algorithm.
        LSID3 is a Lookahead Stochastic variation of ID3 algorithm.
        This algorithm is estimating path size by sampling trees using SID3 algorithm
        and randomly choosing a path.

        Parameters
        ----------
        rValue : int
            A value that represents the lookahead amount.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "LSID3PathSample", 0, rValue)

    def LSID3MC(self, rValue, pValue):
        """
        A Decision Tree wrapper used for LSID3MC algorithm.
        Creating tree with the Monte Carlo evaluation of continuous
        attributes using SID3 algorithm.

        Parameters
        ----------
        rValue : int
            A value that represents the lookahead amount.
        pValue : double
            A value that represents sample size.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "LSID3MC", 0, rValue, pValue)

    def BLSID3(self, rValue):
        """
        A Decision Tree wrapper used for BLSID3 algorithm.
        BLSID3 is a Binary splits variation of LSID3 algorithm.

        Parameters
        ----------
        rValue : int
            A value that represents the lookahead amount.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "LSID3", 1, rValue)

    def BLSID3PathSample(self, rValue):
        """
        A Decision Tree wrapper used for BLSID3PathSample algorithm.
        PathSampleBLSID3 is a Binary splits variation of LSID3PathSample algorithm.

        Parameters
        ----------
        rValue : int
            A value that represents the lookahead amount.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.treeBuilder(sampleIds, attributeIds, self.root, None, "LSID3PathSample", 1, rValue)

    def LSID3Sequenced(self, timeValue):
        """
        A Decision Tree wrapper used for BLSID3 algorithm.
        This is an Interruptible Induction by Sequencing Contract Algorithm.

        Parameters
        ----------
        timeValue : int
            A value that represents the amount of time we want the algorithm
            to run (the value represents seconds).
        """
        
        self.ID3()
        rValue = 1
        startTime = time.time()
        #The iteration will continue until all the time will be utilized
        while True:
            if (time.time() > startTime + timeValue):
                break
            self.LSID3(rValue)
            rValue = 2 * rValue

    def IIDT(self, timeValue, gValue):
        """
        A Decision Tree wrapper used for IIDT algorithm.
        IIDT is an Interruptible Induction by Iterative Improvement.
        This algorithm is an improvement to the LSID3 algorithm.

        Parameters
        ----------
        timeValue : int
            A value that represents the amount of time we want the algorithm
            to run (the value represents seconds).
        gValue : double
            This parameter serves as a threshold for the minimal time
            allocation for an improvement phase.
        """
        
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.createIIDT(sampleIds, attributeIds, timeValue, gValue)
#*****************************END DECISION TREE ALGORITHM WRAPPER******************************#


#*****************************************IIDT METHODS*****************************************#
    def createIIDT(self, sampleIds, attributeIds, timeValue, gValue):
        """
        IIDT is an Interruptible induction of decision trees.
        This algorithm is an improvement to the LSID3 algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        timeValue : int
            A value that represents the amount of time we want the algorithm
            to run (the value represents seconds).
        gValue : double
            This parameter serves as a threshold for the minimal time
            allocation for an improvement phase.

        Returns
        -------
        Node
            A pointer to the tree root. 
        """
        
        self.ID3()
        startTime = time.time()
        #The iteration will continue until all the time will be utilized
        while True:
            if (time.time() > startTime + timeValue):
                break
            chosenNode = self.chooseNodeIIDT(sampleIds, attributeIds, gValue)
            subTreeRoot = chosenNode
            #Calculate all the attributes that are not ancestors of the chosen node
            attributesNodeId = attributeIds.copy()
            nodeAncestors = self.getNodeAncestors(chosenNode)
            #Get the attributes ids
            nodeAncestorsAttId = []
            for ancNode in nodeAncestors:
                nodeAncestorsAttId.append((self.attributes).index(ancNode.value))
            #Gets only attributes that are not ancestors of the chosen node
            for ancId in nodeAncestorsAttId:
                if ancId in attributesNodeId:
                    attributesNodeId.remove(ancId)
            #Get real attributes values
            attributesNode = []
            for attId in attributesNodeId:
                attributesNode.append(self.attributes[attId])
            #Calculate all the samples that reaches node
            eSampleNode = self.sample.copy()
            copySampleIds = sampleIds.copy()
            #Calculate all the sample needed
            childrenSampleIds = []
            for childValue in chosenNode.children:
                for sid in sampleIds:
                    if self.sample[sid][(self.attributes).index(chosenNode.value)] == childValue.value:
                        childrenSampleIds.append(sid)
            for samId in sampleIds:
                if samId not in childrenSampleIds and samId in copySampleIds:
                    copySampleIds.remove(samId)
            #Gets the real values of sample using the needed sample ids
            newLabels = self.labels.copy()
            for sid in sampleIds:
                if sid not in copySampleIds:
                    eSampleNode[sid] = None
                    newLabels[sid] = None
            eSampleNode = [e for e in eSampleNode if e is not None]
            newLabels = [e for e in newLabels if e is not None]
            #Ger r value
            newrValue = self.nextR(chosenNode)
            #Update to the new r value after choosing a node
            chosenNode.lastrValueIIDT = newrValue
            newSubTree = DecisionTree(eSampleNode, attributesNode, newLabels)
            newSubTree.LSID3(newrValue)
            #Evaluate old sub tree with new sub tree using the tree size
            if self.size(subTreeRoot) > newSubTree.size():
                #In case old sub tree size is bigger we replace the old sub tree with the new one
                subTreeRootParent = subTreeRoot.parent
                #If parent is None, the node chosen is the root
                if subTreeRootParent is None:
                    self.root = newSubTree.root
                    self.root.parent = None
                else:
                    #Every node but root
                    subTreeRootParent.next = newSubTree.root
                    newSubTree.parent = subTreeRootParent
        return self.root

    def chooseNodeIIDT(self, sampleIds, attributeIds, gValue):
        """
        Calculate the best Node that need to be selected for the
        IIDT algorithm.

        Parameters
        ----------
        sampleIds : list
            A list of indexs that represents a list of samples from the data.
        attributeIds : list
            A list of indexs that represents a list of attributes from the data.
        gValue : double
            This parameter serves as a threshold for the minimal time
            allocation for an improvement phase.

        Returns
        -------
        Node
            A pointer to the best Node selected by the algorithm. 
        """
        
        maxCost = (self.nextR(self.root) * len(sampleIds) * math.pow(len(attributeIds), 3))
        #Init all Nodes
        listNode = []
        self.getInnerNodesInSubTree(self.root, listNode)
        #Init return values
        maxuNode = -math.inf
        bestNode = self.root
        #Go through all the nodes in a tree
        for testNode in listNode:
            #Calculate all the attributes that are not ancestors of the chosen node
            attributesNodeId = attributeIds.copy()
            nodeAncestors = self.getNodeAncestors(testNode)
            #Get the attributes ids
            nodeAncestorsAttId = []
            for ancNode in nodeAncestors:
                nodeAncestorsAttId.append((self.attributes).index(ancNode.value))
            #Gets only attributes that are not ancestors of the chosen node
            for ancId in nodeAncestorsAttId:
                if ancId in attributesNodeId:
                    attributesNodeId.remove(ancId)
            #Get real attributes values
            attributesNode = []
            for attId in attributesNodeId:
                attributesNode.append(self.attributes[attId])
            #Calculate all the samples that reaches node
            eSampleNode = self.sample.copy()
            copySampleIds = sampleIds.copy()
            #Calculate all the sample needed
            childrenSampleIds = []
            for childValue in testNode.children:
                for sid in sampleIds:
                    if self.sample[sid][(self.attributes).index(testNode.value)] == childValue.value:
                        childrenSampleIds.append(sid)
            for samId in sampleIds:
                if samId not in childrenSampleIds and samId in copySampleIds:
                    copySampleIds.remove(samId)
            #Gets the real values of sample using the needed sample ids
            newLabels = self.labels.copy()
            for sid in sampleIds:
                if sid not in copySampleIds:
                    eSampleNode[sid] = None
                    newLabels[sid] = None
            eSampleNode = [e for e in eSampleNode if e is not None]
            newLabels = [e for e in newLabels if e is not None]
            #Ger r value
            rValueNode = self.nextR(testNode)
            costNode = (rValueNode * len(eSampleNode) * math.pow(len(attributesNode), 3))
            #Check threshold in order to continue calculating for the current test node
            if (costNode/maxCost) > gValue:
                lBound = math.pow(self.getMinDomainAttFromListAtt(sampleIds, attributesNodeId), 2)
                #Init all Leafs
                listLeaf = []
                self.getLeafNodesInSubTree(testNode, listLeaf)
                deltaQ = (len(listLeaf) - lBound)
                uNode = (deltaQ/costNode)
                #Checking for the maximum value
                if uNode > maxuNode:
                    maxuNode = uNode
                    bestNode = testNode
        return bestNode
          
    def nextR(self, currentNode):
        """
        Calculate the next rValue for choosing a node in IIDT algorithm.

        Parameters
        ----------
        currentNode : Node
            A Node class parameter that represents a node in the tree.  

        Returns
        -------
        int
            A value that represents the next r value (the next lookahead value).
        """
        
        if currentNode.lastrValueIIDT == 0:
            return 1
        return (2 * currentNode.lastrValueIIDT)
#***************************************END IIDT METHODS***************************************#
