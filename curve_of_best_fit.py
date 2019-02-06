import numpy as np
import matplotlib.pyplot as plt

#Only supports one feature
class CurveOfBestFit:
    #xs must be 2d numpy array with one row. For ex. [[1, 2, 3]]. ys is
    #a 1d numpy array, for ex. [4, 5, 6]
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    #Returns a 2d numpy array, where each row is a power of xs ie
    #[[1, 2, 3], [1, 4, 9]] etc. depending on the order 
    def _poly(self, order, xs):
        poly_xs = np.ones(xs[0].size)
        for power in range(order):
            poly_xs = np.row_stack((poly_xs, xs**(power+1)))
        return poly_xs

    #Estimates coefficients for training set ie m0 + m1*x1 + m2*x2**2
    def estimate_ms(self, a, order, tolerance):
        np.seterr(all='raise')
        poly_xs = self._poly(order, self.xs)
        train_size = np.size(self.ys)
        ms = np.ones((len(poly_xs),1))
        float_error = True
        while float_error:
            try:
                ms = self._grad_descent(a, ms, poly_xs, self.ys, train_size, tolerance)
                float_error = False
            except FloatingPointError:
                a = a/10
        return ms

    def row_sum(self, arr):
        return arr.sum(axis=0)

    #Partial derivative of the cost function with respect to some coefficient m
    def _part_der_cost(self, a, ms, i, xs, ys, train_size):
        err = (self.row_sum(ms*xs) - ys)*xs[i]
        err_mag = err.sum()
        err_avg = err_mag/train_size
        return err_avg

    #Take a step towards the minima
    def _step(self, a, ms, xs, ys, train_size):
        d_cost_m = np.zeros([len(ms), 1])
        for i in range(len(ms)):
            d = self._part_der_cost(a, ms, i, xs, ys, train_size)
            d_cost_m[i] = d
        ms = ms - a*d_cost_m
        return ms, d_cost_m

    def _grad_descent(self, a, ms, xs, ys, train_size, tolerance):
        ms, d_cost_m = self._step(a, ms, xs, ys, train_size)
        while any(abs(d_cost_m) > tolerance):
            ms, d_cost_m = self._step(a, ms, xs, ys, train_size)
        return ms

def Main():
    xs = np.array([[-2, -1.33, 0, 0.618, 1]])
    ys = np.array([0.5, 1.735, -1, 0.75, 4])
    order = 3
    a = 1
    tolerance = 0.01
    curve = CurveOfBestFit(xs, ys)
    ms = curve.estimate_ms(a, order, tolerance)
    print(ms)
    #Change new_xs as appropriate for plot. Observe if fit generalizes well to new dataset (new_xs)
    new_xs = np.array([np.linspace(-3, 2, 100)])
    new_xs_powered = np.ones(new_xs[0].size)
    for power in range(order):
        new_xs_powered = np.row_stack((new_xs_powered, new_xs**(power+1)))
    plt.plot(curve.row_sum(xs), ys, 'ro')
    plt.plot(curve.row_sum(new_xs), curve.row_sum(new_xs_powered*ms))
    plt.show()

if __name__ == '__main__':
    Main()