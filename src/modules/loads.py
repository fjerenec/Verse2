import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from modules.data import Set, get_user_decision
import numpy as np

class BoundaryCons():
    def __init__(self) -> None:
        pass

class Loads():
    def __init__(self) -> None:
        self.dispLoadsTable = {}
        self.forceLoadsTable = {}

    def create_total_force_load(self, nodeSet: Set, forceX: float, forceY: float, forceZ:float) -> None:
        """ The load values specified are distributed over all points in the defined nodeSet.
        """
        pass

    def create_disp_load(self,nodeSet: Set = None, dispX:float = None, dispY:float = None, dispZ: float = None, name:str ="DispLoad") -> None:
        if name in self.dispLoadsTable.keys():
            print(f"A displacement load with name -{name}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                self.dispLoadsTable[name] = DisplacementLoad(nodeSet=nodeSet, dispX=dispX, dispY=dispY,dispZ=dispZ)
                print("Displacement load overwritten!")
            else:
                return
        else:
            self.dispLoadsTable[name] = DisplacementLoad(nodeSet=nodeSet, dispX=dispX, dispY=dispY,dispZ=dispZ)


class Load():
    def __init__(self) -> None:
        pass

class ForceLoad(Load):
    def __init__(self) -> None:
        super().__init__()

class DisplacementLoad(Load):
    def __init__(self, nodeSet: Set, dispX:float = None, dispY:float = None, dispZ:float = None) -> None:
        super().__init__()
        self.nodeSet = nodeSet
        self.dispX = dispX
        self.dispY = dispY
        self.dispZ = dispZ
        self._create_load_BC_vec()

    def _create_load_BC_vec(self):
        if self.dispX == None and self.dispY == None and self.dispZ == None:
            return print("No DOF was specified as fixed. Displacement load was not created! Specify a displacement in at least one direction to create displacement load")
        
        numOfPtsInSet = self.nodeSet.get_number_of_points()
        self.BC_vec = np.zeros(shape = (numOfPtsInSet,7))
        for i,node_ID in enumerate(self.nodeSet.IDarray):
            self.BC_vec[i,0] = node_ID
            if self.dispX != None:
                self.BC_vec[i,1] = 1.
                self.BC_vec[i,4] = float(self.dispX)
            if self.dispY != None:
                self.BC_vec[i,2] = 1.
                self.BC_vec[i,5] = float(self.dispY)
            if self.dispZ != None:
                self.BC_vec[i,3] = 1.
                self.BC_vec[i,6] = float(self.dispZ)

            
            

        

