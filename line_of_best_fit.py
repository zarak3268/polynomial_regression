import numpy as np
import matplotlib.pyplot as plt

def row_sum(arr):
    return arr.sum(axis=0)

def part_der_cost(a, ms, i, xs, ys, train_size):
    err = (row_sum(ms*xs) - ys)*xs[i]
    err_mag = err.sum()
    err_avg = err_mag/train_size
    return err_avg

def step(a, ms, xs, ys, train_size):
    d_cost_mi = np.zeros([len(ms), 1])
    for i in range(len(ms)):
        d = part_der_cost(a, ms, i, xs, ys, train_size)
        d_cost_mi[i] = d
    ms = ms - a*d_cost_mi
    return ms, d_cost_mi

def grad_descent(a, ms, xs, ys, train_size, tolerance):
    ms, d_cost_mi = step(a, ms, xs, ys, train_size)
    
    while any(abs(d_cost_mi) > tolerance):
        ms, d_cost_mi = step(a, ms, xs, ys, train_size)
    return ms

def Main():
    np.seterr(all='raise')
    base_xs = np.array([[1, 2, 3, 4, 5]])
    ones = np.ones(base_xs[0].size)
    xs = np.row_stack((ones, base_xs))
    ys = np.array([3.6, 4.8, 6.6, 9, 11.8])
    train_size = np.size(ys)
    ms = np.array([[0.5], [0.6]])
    a = 1
    tolerance = 0.001
    float_error = True
    
    while float_error:
        try:
            ms = grad_descent(a, ms, xs, ys, train_size, tolerance)
            float_error = False
        except FloatingPointError:
            a = a/10
    
    print(ms)
    plt.plot(row_sum(base_xs), ys, 'ro')
    plt.plot(row_sum(base_xs), row_sum(xs*ms))
    plt.show()

if __name__ == '__main__':
    Main()