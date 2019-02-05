import numpy as np
import matplotlib.pyplot as plt

def row_sum(arr):
    return arr.sum(axis=0)

#Partial derivative of the cost function with respect to some coefficient m
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
    #Change base_xs and ys for new data. Change order to obtain different polynomials for fit.
    #a is the learning coefficient
    order = 3
    base_xs = np.array([[-2, -1.33, 0, 0.618, 1]])
    xs = np.ones(base_xs[0].size)
    for power in range(order):
        xs = np.row_stack((xs, base_xs**(power+1)))
    ys = np.array([0.5, 1.735, -1, 0.75, 4])
    train_size = np.size(ys)
    ms = np.ones((len(xs),1))
    a = 1
    tolerance = 0.01
    float_error = True
    
    while float_error:
        try:
            ms = grad_descent(a, ms, xs, ys, train_size, tolerance)
            float_error = False
        except FloatingPointError:
            a = a/10
    
    print(ms)
    #Change new_xs as appropriate for plot. Observe if fit generalizes well to new dataset (new_xs)
    new_xs = np.array([np.linspace(-3, 2, 100)])
    new_xs_powered = np.ones(new_xs[0].size)
    for power in range(order):
        new_xs_powered = np.row_stack((new_xs_powered, new_xs**(power+1)))
    plt.plot(row_sum(base_xs), ys, 'ro')
    plt.plot(row_sum(new_xs), row_sum(new_xs_powered*ms))
    plt.show()

if __name__ == '__main__':
    Main()