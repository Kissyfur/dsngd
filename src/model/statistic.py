import numpy as np


class Statistic:
    def __init__(self, dim):
        self.dim = dim

    def transform(self, rv):
        return rv


class CanonicalStatistic(Statistic):
    def __init__(self, many_values):
        self.many_values = many_values
        super().__init__(many_values - 1)

    def identity(self, arr):
        arr = arr.copy().astype(int)
        id_arr = np.identity(self.many_values, dtype=int)
        id_arr = id_arr[arr]
        return id_arr

    def k(self, arr):
        arr = arr.copy().astype(int)
        id_arr = np.identity(self.many_values, dtype=int)
        id_arr[-1, :] = -1
        id_arr = id_arr[arr]
        return id_arr

    def canonical_statistic(self, arr):
        c_arr = self.identity(arr)
        return c_arr[:, :-1]

    def tau(self, arr):
        k_arr = self.k(arr)
        return k_arr[:, :-1]

    def transform(self, rv):
        return self.canonical_statistic(rv)


class CanonicalFeatureStatistic(Statistic):
    def __init__(self, features):
        self.canonical_statistics = [CanonicalStatistic(feature) for feature in features]
        self.m = features
        self.r = len(features)
        dim = np.sum([cs.dim for cs in self.canonical_statistics])
        self.psi = np.prod(self.m)
        super().__init__(dim)

    def transform(self, rvs):
        return np.hstack([cs.transform(rv) for cs, rv in zip(self.canonical_statistics, rvs.T)])

    def identity(self, rvs):
        return np.hstack([cs.identity(rv) for cs, rv in zip(self.canonical_statistics, rvs.T)])



if __name__ == "__main__":
    T = CanonicalStatistic(3)
    x = np.array([0])
    print(T.tau(x).T)