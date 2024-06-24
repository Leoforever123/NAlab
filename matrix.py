import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# 列主元素高斯消去法解方程
def Gause_solve(A, b):
    n = b.size
    for i in range(0, n):
        # 找主元素
        Max = A[i][i]
        Max_index = i
        for j in range(i, n):
            if (A[j][i] > Max):
                Max = A[j][i]
                Max_index = j
        
        # 交换主元素行和当前行
        for j in range(i, n):
            t = A[i][j]
            A[i][j] = A[Max_index][j]
            A[Max_index][j] = t
        t = b[i]
        b[i] = b[Max_index]
        b[Max_index] = t
        
        if (A[i][i] == 0):
            print("It cannot be solved!")
            return -1
        # 消去过程
        for j in range(i + 1, n):
            f = A[j][i] / A[i][i]
            for s in range(i, n):
                A[j][s] -=  f * A[i][s]
            b[j] -= f * b[i]

    # solve
    x = np.zeros(n)
    index = n - 1
    while(index >= 0):
        temp = b[index]
        for j in range(index + 1, n):
            temp -= A[index][j] * x[j]
        x[index] = temp / A[index][index]
        index -= 1

    return x