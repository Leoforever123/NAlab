import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def polynomial_fit(x, y, degree):
    # Fit the polynomial
    coefs = np.polyfit(x, y, degree)
    poly = np.poly1d(coefs)
    return poly

def plot_fit(x, y, poly, degree):
    # Generate x values for plotting the polynomial
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit = poly(x_fit)
    
    # Plot the original data points
    plt.scatter(x, y, color='red', label='Data points')
    
    # Plot the polynomial fit
    plt.plot(x_fit, y_fit, label=f'{degree} degree polynomial fit')

    # Add title and labels
    plt.title(f'{degree} Degree Polynomial Fit')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('images\degree' + str(degree) + '.png')
    plt.show()

def plot_combined_fit(x, y, poly3, poly4):
    # Generate x values for plotting the polynomial
    x_fit = np.linspace(min(x), max(x), 1000)
    y_fit3 = poly3(x_fit)
    y_fit4 = poly4(x_fit)
    
    # Plot the original data points
    plt.scatter(x, y, color='red', label='Data points')
    
    # Plot the polynomial fits
    plt.plot(x_fit, y_fit3, label='3rd degree polynomial fit')
    plt.plot(x_fit, y_fit4, label='4th degree polynomial fit')

    # Add title and labels
    plt.title('3rd and 4th Degree Polynomial Fits')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('images\combined_fit.png')
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
    poly3 = polynomial_fit(x, y, 3)
    plot_fit(x, y, poly3, 3)
    
    # Fit and plot 4th degree polynomial
    poly4 = polynomial_fit(x, y, 4)
    plot_fit(x, y, poly4, 4)

    # Plot the fits together in one graph
    plot_combined_fit(x, y, poly3, poly4)

if __name__ == "__main__":
    main()
