import matplotlib.pyplot as plt

class DecisionTreePlot:
    def __init__(self):
        """
        The Constructor for DecisionTreePlot class.
        """
                
        pass
                
    def createDecisionTreeNode(self, decisionTree, currentNode, parentPoint, middleTextValue):
        """
        Create the design and gets the value of each node in the Decision Tree.
        
        Parameters
        ----------
        decisionTree : DecisionTree
            A DecisionTree class parameter that represents the Decision Tree.
        currentNode : Node
            A Node class parameter that represents a node in the tree.
        parentPoint : list
            A List with to values that represents the x and y values of a point.
        middleTextValue : string
            A value that represents the text value of the attribute.
        """
                
        #Calculate node center point
        centerPoint = (DecisionTreePlot.createDecisionTreeNode.xOff + (1.0 + float(decisionTree.getNumLeafs(currentNode)))/2.0/DecisionTreePlot.createDecisionTreeNode.totalW,
                       DecisionTreePlot.createDecisionTreeNode.yOff)
        #Create decision tree plot node
        if currentNode.children:
            boxType = dict(boxstyle = "square", fc = "0.7")
        else: #In case the node is a leaf we change the design
            boxType = dict(boxstyle = "round4", fc = "w")
        DecisionTreePlot.createDecisionTreePlot.ax1.annotate(currentNode.value, xy = parentPoint, xycoords = "axes fraction",
                                         xytext = centerPoint, textcoords = "axes fraction", va = "bottom",
                                         ha = "center", bbox = boxType, arrowprops = dict(arrowstyle = "<|-", facecolor = "black"))
        if currentNode is not None:
            #Create decision tree plot node lable
            middlePointX = (parentPoint[0] - centerPoint[0])/2.0 + centerPoint[0]
            middlePointY = (parentPoint[1] - centerPoint[1])/2.0 + centerPoint[1]
            DecisionTreePlot.createDecisionTreePlot.ax1.text(middlePointX, middlePointY, middleTextValue, va = "center", ha = "center", rotation = 30)
        secondDict = currentNode.children
        DecisionTreePlot.createDecisionTreeNode.yOff = DecisionTreePlot.createDecisionTreeNode.yOff - 1.0/DecisionTreePlot.createDecisionTreeNode.totalD
        #Create the node only for non leaf nodes
        if secondDict:
            #For each feature, we create another node for the next attribute
            for child in secondDict:
                DecisionTreePlot.createDecisionTreeNode.xOff = DecisionTreePlot.createDecisionTreeNode.xOff + 0.5/DecisionTreePlot.createDecisionTreeNode.totalW
                self.createDecisionTreeNode(decisionTree, child.next, centerPoint, str(child.value))
        DecisionTreePlot.createDecisionTreeNode.yOff = DecisionTreePlot.createDecisionTreeNode.yOff + 1.0/DecisionTreePlot.createDecisionTreeNode.totalD

    def createDecisionTreePlot(self, decisionTreeCurrentNode):
        """
        Create the Decision Tree plot.
        
        Parameters
        ----------
        decisionTreeCurrentNode : Node
            A Node class parameter that represents a node in the tree.
        """
        
        fig = plt.figure(1, facecolor = 'white', figsize = (80, 30))
        fig.clf()
        axprops = dict(xticks = [], yticks = [])
        #Calculate the position of each node
        DecisionTreePlot.createDecisionTreePlot.ax1 = plt.subplot(111, frameon = False, **axprops)
        DecisionTreePlot.createDecisionTreeNode.totalW = float(decisionTreeCurrentNode.getNumLeafs())
        DecisionTreePlot.createDecisionTreeNode.totalD = float(decisionTreeCurrentNode.getTreeDepth())
        DecisionTreePlot.createDecisionTreeNode.xOff = -0.5/self.createDecisionTreeNode.totalW;
        DecisionTreePlot.createDecisionTreeNode.yOff = 1.0
        self.createDecisionTreeNode(decisionTreeCurrentNode, decisionTreeCurrentNode.root, (0.5, 1.0), middleTextValue = '')
        plt.show()

