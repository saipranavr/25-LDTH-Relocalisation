
from enum import Enum
from dataclasses import dataclass

class Projection(Enum):
    EPSG_4326 = "EPSG:4326"
    EPSG_3035 = "EPSG:3035"

@dataclass
class BoundingBox:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float  
    projection: Projection

    def to_query_string(self) -> str:
        return f"{self.min_lon},{self.min_lat},{self.max_lon},{self.max_lat}"