import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#Just a line function.
def lin_func(ms, xs):
    return ms*xs

#Step down the cost function towards the minima
def step(a, ms, xs, ys, sample_size):
    d_cost = np.zeros([ms.size, 1])
    for i in range(len(ms)):
        d = part_deriv_cost_m(i, ms, xs, ys, sample_size)
        d_cost[i] = d
    ms = ms - a*d_cost
    return ms, d_cost

#a is the learning rate
def grad_descent(a, ms, xs, ys, sample_size):
    ms, d_cost = step(a, ms, xs, ys, sample_size)
    all_d_cost_is_zero = all(abs(d) < 0.01 for d in d_cost)
    while not all_d_cost_is_zero:
        ms, d_cost = step(a, ms, xs, ys, sample_size)
        all_d_cost_is_zero = all(abs(d) < 0.01 for d in d_cost)
    return ms

#Derivative of the cost function with respect to ith coeff m
def part_deriv_cost_m(i, ms, xs, ys, sample_size):
    #err = (ms*xs - ys)*xs[i]
    err = ms*xs
    err = (row_sum(err) - ys)*xs[i]
    err_mag = err.sum()
    result = err_mag/(sample_size)
    return result

def row_sum(arr):
    return arr.sum(axis=0)

def Main():
    #Replace xs and ys with desired x and y coordinates
    #Change a to optimal value. Lower it if getting Runtimewarning
    xs = np.array([[1, 2, 3, 4, 5]
                 ,[1, 2, 3, 4, 5]])
    ys = np.array([1, 2, 3, 4, 5])
    ms = np.array([[0.6], [0.5]])
    sample_size = ys[0].size
    a = 0.01
    ms = grad_descent(a, ms, xs, ys, sample_size)
    print(ms)
    ax = plt.axes(projection='3d')
    output = row_sum(ms*xs)
    ax.plot3D(output, xs[0], xs[1],'gray')
    ax.plot3D(ys, xs[0], xs[1],'ro')
    
if __name__ == '__main__':
    Main()