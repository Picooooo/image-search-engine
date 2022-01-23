import numpy as np
from scipy.spatial import distance

class Euclidean:

    def similarity_metric(self, X, Y):
        return distance.euclidean(X, Y)