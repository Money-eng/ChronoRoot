from openalea.mtg import MTG
from rsml.misc import plant_vertices

from ..base import BaseMeasure


class NumberOfPlants(BaseMeasure):
    type = "cpu"
    need = "serie"

    def __init__(self):
        super().__init__()

    def __call__(self, mtg: MTG) -> float:
        return len(plant_vertices(mtg))
