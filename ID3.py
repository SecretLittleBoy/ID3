import math
import operator
import TreePlotter


def createDataset() -> tuple[list[list[str]], list[str]]:
    dataSet = [
        ['T', 'T', '+'],
        ['T', 'T', '+'],
        ['T', 'F', '-'],
        ['F', 'F', '+'],
        ['F', 'T', '-'],
        ['F', 'T', '-']
    ]
    # 特征值列表
    labels = ['x1', 'x2']
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    shannonEnt = -p1*log(p1, 2)-p2*log(p2, 2)-...-pn*log(pn, 2)
    p1, p2, ..., pn为数据集中不同类别的概率
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1

    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*math.log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet: list[list[str]], axis: int, value: str) -> list[list[str]]:
    '''
    remove the axis-th feature of the dataSet
    only keep the rows that axis-th feature is equal to value
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet: list[list[str]]) -> int:
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        uniqueFeatList = set([example[i] for example in dataSet])
        newEntropy = 0

        for value in uniqueFeatList:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majority(classList: list[str]) -> str:
    '''
    返回list[str]中出现次数最多的str
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet: list[list[str]], labels: list[str]):
    classList = [example[-1] for example in dataSet]
    # 如果类别完全相同，则停止继续划分
    if classList.count(classList[0]) == len(dataSet):
        return classList[0]
    # 如果没有特征可以划分，则停止继续划分
    if len(labels) == 0:
        return majority(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    del (labels[bestFeat])

    myTree = {bestFeatLabel: {}}
    uniquefeatVals = set([example[bestFeat] for example in dataSet])
    for value in uniquefeatVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myTree = createTree(createDataset()[0], createDataset()[1])
print(myTree)
TreePlotter.createPlot(myTree)
