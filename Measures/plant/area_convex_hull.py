import numpy as np
from openalea.mtg import MTG
from scipy.spatial import ConvexHull

from ..base import BaseMeasure


def convex_hull_area(points):
    points = np.array(points)
    hull = ConvexHull(points)
    return hull.area


class Convex_Area_Hull(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        geometry = mtg.property('geometry')  # {1: [[598.0, 148.0], [597.0, 162.0], ...]}

        points = [point for points in geometry.values() for point in points]

        ch_area = convex_hull_area(points)
        return ch_area
