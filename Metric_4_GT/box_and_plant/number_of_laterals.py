from openalea.mtg import MTG
from rsml.misc import root_vertices as rt_verts

from ..base import BaseMeasure


class NumberOfLateralRoots(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        root_vertices = rt_verts(mtg)
        # if no parents, remove from root vertices
        lateral_vertices = [v for v in root_vertices if mtg.parent(v) is not None]
        return len(lateral_vertices) if lateral_vertices else 0.0
