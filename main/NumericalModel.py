import geometry
import material_model
import disretization
import boundary_cons
import interactions
import simulation_settings
import data
from typing import IO

class NumericalModel():
    def __init__(self,name: str = "Model") -> None:
        self.name = name
        self.Geometry = geometry.Geometry()
        self.MaterialModel = material_model.MaterialModel()
        self.Discretization = disretization.Discretization()
        self.BoundaryCons = boundary_cons.BoundaryCons()
        self.Interactions = interactions.Interactions()
        self.SimulationSettings = simulation_settings.SimulationSettings()
    
    def createInputFile(self) -> IO[str]:
        pass