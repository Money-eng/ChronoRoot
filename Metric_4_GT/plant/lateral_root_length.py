from openalea.mtg import MTG

from ..base import BaseMeasure


def lateral_root_length(mtg: MTG) -> float:
    roots = mtg.vertices(scale=mtg.max_scale())
    # remove 1st vertex 
    roots.pop(0)
    total_length = 0.0
    for root in roots:
        geometry = mtg.property("geometry")
        polyline = geometry[root]
        for i in range(len(polyline) - 1):
            length = ((polyline[i][0] - polyline[i + 1][0]) ** 2 +
                      (polyline[i][1] - polyline[i + 1][1]) ** 2) ** 0.5
            total_length += length
    return total_length


class LateralRootLength(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return lateral_root_length(mtg)
