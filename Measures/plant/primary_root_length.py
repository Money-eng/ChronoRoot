from openalea.mtg import MTG

from ..base import BaseMeasure


def primary_root_length(mtg: MTG) -> float:
    roots = mtg.vertices(scale=mtg.max_scale())
    root = roots[0]
    geometry = mtg.property("geometry")
    polyline = geometry[root]
    total_length = 0.0
    for i in range(len(polyline) - 1):
        length = ((polyline[i][0] - polyline[i + 1][0]) ** 2 +
                  (polyline[i][1] - polyline[i + 1][1]) ** 2) ** 0.5
        total_length += length
    return total_length


class PrimaryRootLength(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return primary_root_length(mtg)
