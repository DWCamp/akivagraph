import math


def dist(a, b):
    """
    Computes the Euclidean distance between two points
    :param a: The (x, y) coordinates of point a
    :param b: The (x, y) coordinates of point b
    :return: The Euclidean distance between them
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
