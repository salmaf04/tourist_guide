from typing import Optional, Union
from attr import dataclass

@dataclass
class TouristPlace:
    name: str
    city: str
    category: str
    description: str    
    coordinates: Optional[Union[tuple, None]]