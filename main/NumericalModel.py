import geometry
import material
import disretization
import loads
import interactions
import simulation_settings
import data
from typing import IO

class NumericalModel():
    def __init__(self,name: str = "Model") -> None:
        self.name = name
        self.Geometry = geometry.Geometry()
        self.Materials= material.Materials()
        self.MaterialSections = material.MaterialSections()
        self.MaterialInterfaces = material.MaterialInterfaces()
        self.Loads = loads.Loads()
        self.Discretizations = disretization.Discretizations()
        self.Interactions = interactions.Interactions()
        self.SimulationSettings = simulation_settings.SimulationSettings()
    
    def createInputFile(self) -> IO[str]:
        pass

    