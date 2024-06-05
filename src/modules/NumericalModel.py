import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import modules.geometry as geometry
import modules.material as material 
import modules.disretization as disretization
import modules.loads as loads
import modules.interactions as interactions
import modules.simulation_settings as simulation_settings
import modules.data as data
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

    