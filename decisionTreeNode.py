
class DecisionTreeNode(object):
    def __init__(self):
        """
        The Constructor for a Node in the DecisionTree class.
        """
        
        self.value = None #Node value (can be any type of value)
        self.next = None #Pointer to the next attribute if exists
        self.children = None #Features of the current Node attribute
        self.lastrValueIIDT = 0 #Last r value (lookahead value) used in the IIDT algorithm
        self.parent = None #Parent of the Node
        self.majorityClass = None #Used for pruning. This is the label that appears the most when we choose a path from the root to the node
        self.errorValue = None #Used for pruning. This is the number of labels that are not the majority class (label)
