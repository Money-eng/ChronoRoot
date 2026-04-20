from openalea.mtg import MTG

from ..base import BaseMeasure

from rsml.misc import root_vertices

def total_root_length(mtg: MTG) -> float:
    roots = root_vertices(mtg)
    total_length = 0.0
    for root in roots:
        geometry = mtg.property("geometry")
        polyline = geometry[root]
        for i in range(len(polyline) - 1):
            length = ((polyline[i][0] - polyline[i + 1][0]) ** 2 +
                      (polyline[i][1] - polyline[i + 1][1]) ** 2) ** 0.5
            total_length += length
    return total_length

class TotalRootLength(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return total_root_length(mtg)
