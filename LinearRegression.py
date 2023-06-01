import csv
# import pandas as pd
# import numpy as np

def generateEquation():
    dataList = []
    dataListLength = 0
    currentX = 0
    currentY = 0
    xSquraedSumnation = 0
    xySumnation = 0
    dataListLength = 0

    with open('myCsv.csv', mode='r') as data:
        reader = csv.reader(data)
        for row in reader:
            dataList.append(row)
            dataListLength += 1
    for step in range(dataListLength):
        # print(step)
        xSquraedSumnation += float(dataList[step][0]) * float(dataList[step][0])
        # print(xSquraedSumnation)
        currentX += float(dataList[step][0])
        currentY += float(dataList[step][1])

        localX = float(dataList[step][0])
        localY = float(dataList[step][1])
        xySumnation += localX * localY
    xSquaredAfterSumnation = currentX**2
    # m = covariance / variance
    m = (dataListLength * xySumnation - currentX * currentY) / (dataListLength * xSquraedSumnation - xSquaredAfterSumnation) 
    b = (currentY - m * currentX) / dataListLength
    print("The equation: " + str(m) + "x" + " + " + str(b))

generateEquation()