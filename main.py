import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import matrix as mx

# alpha * e ** (beta * x)
def exp_constructor(alpha, beta):
    def f(x):
        return alpha * np.exp(beta * x)
    return f

def delta(x, y, poly):
    return np.sum((y - poly(x))**2)

def exp_fit(x, y):
    # Construct the matrix A and vector b
    A = np.zeros((2, 2))
    b = np.zeros(2)
    y = np.log(y)
    
    A[0][0] = x.size
    A[0][1] = np.sum(x)
    A[1][0] = A[0][1]
    A[1][1] = np.sum(x**2)
    b[0] = np.sum(y)
    b[1] = np.sum(x * y)
    
    # Solve for the coefficients
    coefs = mx.Gause_solve(A, b)
    
    # Construct the exponential function
    alpha = np.exp(coefs[0])
    beta = coefs[1]
    f = exp_constructor(alpha, beta)
    
    return f

def poly_fit(x, y, degree):
    m = x.size
    f = []
    n = degree + 1
    for i in range(0, 2 * degree + 1):
        temp = 0
        for j in range(0, m):
            temp += x[j] ** i
        f.append(temp)
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = f[i + j]

    b = np.zeros(n)
    for i in range(0, n):
        temp = 0
        for j in range(0, m):
            temp += y[j] * (x[j] ** i)
        b[i] = temp

    coefs = mx.Gause_solve(A, b)
    coefs = coefs[::-1]
    poly = np.poly1d(coefs)

    return poly

def plot_combined_fit(x, y, functions):
    # Generate x values for plotting the polynomial
    x_fit = np.linspace(min(x), max(x), 1000)
    for f in functions:
        y_fit = f[0](x_fit)
        plt.plot(x_fit, y_fit, label = f[1])
    
    # Plot the original data points
    plt.scatter(x, y, color='red', label='Data points')

    # Add title and labels
    plt.title('3rd, 4th Degree Polynomial Fits and my_exp Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('images/combined_fit.png')
    plt.show()

def main():
    # Input number of data points
    num_points = int(input("请输入离散数据点的数量："))
    
    # Input data points
    data_points = []
    for i in range(num_points):
        point = input(f"请输入第{i+1}对离散数据点（格式：x,y）：")
        data_points.append(point)
    
    x = []
    y = []
    for point in data_points:
        xi, yi = map(float, point.split(','))
        x.append(xi)
        y.append(yi)
    
    x = np.array(x)
    y = np.array(y)
    
    # Fit and plot 3rd degree polynomial
    poly3 = poly_fit(x, y, 3)
    print("3次多项式拟合误差：{:6f}".format(delta(x, y, poly3)))

    # Fit and plot 4th degree polynomial
    poly4 = poly_fit(x, y, 4)
    print("4次多项式拟合误差：{:6f}".format(delta(x, y, poly4)))

    exp_f = exp_fit(x, y)
    print("指数函数拟合误差：{:6f}".format(delta(x, y, exp_f)))

    # 去掉第一个点的指数函数拟合
    x_2 = x[1:]
    y_2 = y[1:]
    exp_f_2 = exp_fit(x_2, y_2)
    print("去掉首点的指数函数拟合误差：{:6f}".format(delta(x_2, y_2, exp_f_2)))
    
    functions = [[poly3, '3rd degree polynomial fit'], 
                 [poly4, '4th degree polynomial fit'], 
                 [exp_f, 'exp fit'],
                 [exp_f_2, 'exp fit without the first point']
                 ]
    
    # Plot the fits together in one graph
    plot_combined_fit(x, y, functions)

    

if __name__ == "__main__":
    main()
