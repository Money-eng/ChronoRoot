# Metrics/__init__.py

from .base import BaseMeasure
from .box.number_of_plants import NumberOfPlants
from .box_and_plant.number_of_laterals import NumberOfLateralRoots
from .box_and_plant.number_of_organs import NumberOfOrgans
from .box_and_plant.total_root_length import TotalRootLength
from .plant.area_convex_hull import Convex_Area_Hull
from .plant.lateral_root_length import LateralRootLength
from .plant.primary_root_length import PrimaryRootLength
from .plant.root_density import RootDensity


# Global dictionnary to map measure names to their corresponding classes
MEASURES_FACTORIES = {
    # Per box
    "number_of_plants": NumberOfPlants,
    # Per box and plant
    "number_of_organs": NumberOfOrgans,
    "total_root_length": TotalRootLength,
    "number_of_laterals": NumberOfLateralRoots,
    # Per plant
    "convex_area_hull": Convex_Area_Hull,
    "root_density": RootDensity,
    "primary_root_length": PrimaryRootLength,
    "lateral_root_length": LateralRootLength,
}


def get_measure(metric_config: dict) -> BaseMeasure:
    """
    Instanciate a given metric based on its configuration.
    {
        "name": "dice",
        "params": {...} 
    }
    """
    name = metric_config["name"]
    params = metric_config.get("params", {})
    if name not in MEASURES_FACTORIES:
        raise ValueError(
            f"Unknown metric: {name}. Known: {list(MEASURES_FACTORIES.keys())}")
    try:
        return MEASURES_FACTORIES[name](**params)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating metric '{name}' with params {params}: {e}")


def get_measures(metrics_config: dict) -> dict:
    result = {"per_plant": [], "per_box": []}
    for t in ["per_plant", "per_box"]:
        for cfg in metrics_config.get(t, []):
            metric = get_measure(cfg)
            result[t].append(metric)
    return result
