# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from R.utils.linearTransformation import linearTransformation


class bounds(object):
    """
    Class bounds. Bounding Hyper-rectangle.
    """

    def __init__(self, min, max):
        """
        
        :param min: list or array  
        :param max: list or array
        """
        self.min = np.array(min)
        self.max = np.array(max)
        self.n_dim = self.min.size
        assert self.min.shape == self.max.shape
        assert np.any(self.min < self.max)

    def get_mean(self, idx=None):
        if idx is None:
            return self.min + (self.max-self.min)/2
        else:
            assert np.all(idx < self.n_dim), 'invalid idx'
            if self.n_dim == 1:  # This is necessary because of the inconstistent behavior of numpy
                return self.min + (self.max - self.min) / 2
            else:
                return self.min[idx] + (self.max[idx]-self.min[idx])/2

    def get_min(self, idx=None):
        """
        Return the lower bounds of all the dimensions, or only the specified one
        :param idx:
        :return:
        """
        if idx is None:
            return self.min
        else:
            assert np.all(idx < self.n_dim), 'invalid idx'
            if self.n_dim == 1:  # This is necessary because of the inconstistent behavior of numpy
                return self.min
            else:
                return self.min[idx]

    def get_max(self, idx=None):
        """
        Return the upper bounds of all the dimensions, or only the specified one
        :param idx:
        :return:
        """
        if idx is None:
            return self.max
        else:
            assert np.all(idx < self.n_dim), 'invalid idx'
            if self.n_dim == 1:  # This is necessary because of the inconstistent behavior of numpy
                return self.max
            else:
                return self.max[idx]

    def get_both(self, idx=None):
        """
        
        :param idx: 
        :return: 
        """
        return np.vstack([self.get_min(idx=idx), self.get_max(idx=idx)])

    def get_delta(self, idx=None):
        if idx is None:
            return self.max-self.min
        else:
            # TODO implement me
            pass

    def to_list(self, idx=None):
        """
        export the bounds to a list formatted as
        [[min_1, min_2 ... min_n][max_1, max_2, ... max_n]]
        :param idx: 
        :return: 
        """
        return [self.get_min(idx=idx), self.get_max(idx=idx)]

    def to_scipy(self, idx=None):
        """
        export the bounds to the scipy.minimize format
        ((min_1, max_1), ... (min_n, max_n))
        :param idx: 
        :return: 
        """
        x = []
        if idx is not None:
            d = idx
        else:
            d = self.n_dim
        for i in range(d):
            x.append((self.get_min(idx=i), self.get_max(idx=i)))
        return tuple(x)

    def get_n_dim(self):
        """
        Return the dimensionality of the bounds
        :return:
        """
        return self.n_dim

    def get_corner_points(self):
        """
        Return the list of corners of the hyper-rectangle 
        :return: 
        """

        def cartesian2(arrays):
            arrays = [np.asarray(a) for a in arrays]
            shape = (len(x) for x in arrays)

            ix = np.indices(shape, dtype=int)
            ix = ix.reshape(len(arrays), -1).T

            for n, arr in enumerate(arrays):
                ix[:, n] = arrays[n][ix[:, n]]

            return ix
        l = []
        for i in range(self.get_n_dim()):
            l.append(self.get_both(idx=i)[:,0])
        y = cartesian2(l)
        return y

    def sample_uniform(self, n):
        """
        Sample uniformly from the bounding hyperrectangle
        :param n: tuple
        :return: ndarray  self.n_DIM x n where n can be a tuple. e.g., n=(2, 4) => self.n_DIM x 2 x 4
        """
        # t = n + self.n_dim
        # return np.swapaxes((self.get_min() + (self.get_max() - self.get_min()) * np.random.rand(*t)), 0, -1).squeeze()
        return self.get_min() + (self.get_max() - self.get_min()) * np.random.rand(*n)
        # TODO: make code more robust by considering self.n_dim explicitly

    def isWithinBounds(self, data):
        """

        :param data: np.array
        :return:
        """
        assert data.ndim == 1
        return np.all(self.get_min() <= data) and np.all(data <= self.get_max())

    def transform(self, new_bounds):
        """
        Compute the linear transformation that map the current bounds, to a new set of bounds
        :param new_bounds: new set of bounds
        :return: linear transformation
        """
        A = np.diag(new_bounds.get_delta()/self.get_delta())
        B = - self.get_min()*new_bounds.get_delta()/self.get_delta() + new_bounds.get_min()
        transformation = linearTransformation(A=A, B=B)
        return transformation

    def transform_01(self):
        """
        Compute the linear transformation to the bounds [0,1]
        :param new_bounds:
        :return:
        """
        newbounds = bounds(min=[0] * self.get_n_dim(), max=[1] * self.get_n_dim())
        return self.transform(newbounds)  # Compute bounds transformation
        # TODO: test this function!!!

    def project2Bounds(self, x):
        """
        Project a point outside the bounds to the bounds minimizing the norm 2 distance
        :param x: []
        :return: 
        """
        # TODO: vectorize
        x = np.matrix(x)
        if self.isWithinBounds(np.array(x).flatten()):
            return x  # nothing to do here, is already within the bounds
        else:
            out = np.multiply(x, np.logical_and(self.get_min() <= x, x <= self.get_max())) + \
                  np.multiply(self.get_min(), (self.get_min() > x)) + \
                  np.multiply(self.get_max(), (x > self.get_max()))
            assert self.isWithinBounds(np.array(out).flatten()), 'Something wrong!'
            return out

    def subset(self, idx):
        """
        Create a bounds object with only a subset of the dimensions
        :param idx:
        :return:
        """
        idx = np.array(idx)
        subset_min = self.get_min(idx)
        subset_max = self.get_max()
        subset = bounds(min=subset_min, max=subset_max)
        return subset

# if __name__ == '__main__':
#     min = [-1, -2, -3]
#     max = [1, 2, 3]
#     a = bounds(min=min, max=max)
#     out = a.get_corner_points()
