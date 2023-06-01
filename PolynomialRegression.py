import csv
import numpy as np

def poly(csvFile, degree):

    # dataListLength = n
    n = 0

    # m = degree
    m = degree

    # lists
    dataList = []
    leftPowerAry = np.zeros(2*m + 1)
    rightPowerAry = np.zeros(m + 1)
    leftMatrix = np.zeros((m+1, m+1))
    rightMatrix = np.zeros((m+1, 1))

    with open(csvFile, mode="r") as data:
        reader = csv.reader(data)
        for row in reader:
            dataList.append(row)
            n += 1

    # Adds the combined value of each x column to an array, each value in the array being a different column (1)
    for power in range(2*m + 1):
        for i in range(n):
            leftPowerAry[power] += float(dataList[i][0]) ** power

    for power in range(m + 1):
        for i in range(n):
            rightPowerAry[power] += float(dataList[i][1]) * float(dataList[i][0]) ** power

    for row in range(m+1):
        for col in range(m+1):
            leftMatrix[row][col] = leftPowerAry[m*2-col-row]

    for row in range(m+1):
        rightMatrix[row][0] = rightPowerAry[m-row] 

    coefAry = np.linalg.solve(leftMatrix, rightMatrix)
    for coef in coefAry:
        print(coef)

poly("myCsv.csv", 3)
