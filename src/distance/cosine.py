import numpy as np
from scipy.spatial import distance

class Cosine:
    def similarity_metric(self, X, Y):
        return 1 - distance.cosine(X, Y)