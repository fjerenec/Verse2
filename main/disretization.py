import geometry
import numpy as np
import pddopyW2 as pddo
from data import get_user_decision

class Discretizations():
    def __init__(self) -> None:
        self.discretizationsTable = {}


    def create_discretization(self,discName:str = "Discretization Name"):
        if type(discName) != str:
            discName = str(discName)

        if discName in self.discretizationsTable.keys():
            print(f"A discretization with name -{discName}- already exists. Would you like to overwrite?")
            override = get_user_decision()
            if override == True:
                print("Discretization overwritten")
                self.discretizationsTable[discName] = Discretization()
            else:
                return
        else:
            self.discretizationsTable[discName] = Discretization()

    def set_selected_discretization(self,DiscretizationName: str) -> None:
        if DiscretizationName in self.discretizationsTable.keys():
            self.selectedDiscretization = self.discretizationsTable[DiscretizationName]
        else:
            print("No discretization with given name. Please provide a valid name!")

class Discretization():

    def __init__(self) -> None:
        self.hasInitialCrack = False
        self.initialCracks = np.empty(shape=(0,2,2))

    def set_horizon(self, horizonRad) -> None:
        self.delta = float(horizonRad)

    def create_crack(self,p1x,p1y,p2x,p2y) -> None:
        crack = np.array([[[p1x,p1y],[p2x,p2y]]],dtype=float)
        self.initialCracks.append(crack)
        self.hasInitialCrack = True

    def deactivate_cracks(self) -> bool:
        self.hasInitialCrack = False

    def activate_cracks(self) -> bool:
        self.hasInitialCrack = True

    def is_crack_active(self) -> bool:
        return self.hasInitialCrack

    def get_node_family_IDs(self,nodeID: int) -> np.ndarray[int,1]:
        nodeIdPosition = np.where(self.nodeIdIndeces == nodeID)
        firstMemberIndex = self.start_idx[nodeIdPosition]
        lastMemberIndex = self.end_idx[nodeIdPosition]
        nodeFamily = self.nodeIdIndeces[self.neighbors[firstMemberIndex:lastMemberIndex]]
        return nodeFamily

    def get_node_family_coords(self,nodeID: int) -> np.ndarray[int,2]:
        firstMemberIndex = self.start_idx[nodeID]
        lastMemberIndex = self.end_idx[nodeID]
        nodeFamilyCoords = self.coordVec[self.neighbors[firstMemberIndex:lastMemberIndex]]
        return nodeFamilyCoords


    def generate_bonds(self,partNodes: geometry._PartNodes) -> None:
        #Check if everything needed for this function is satisfied!


        #Save the IDs of each node to an array (deletions and additions of nodes -> node id might not be sequential -> save the ID in the sequence they are in in the partNodesTable)
        self.nodeIdIndeces = np.zeros(shape=(len(partNodes.partNodesTable)))
        self.coordVec = np.zeros(shape=(len(partNodes.partNodesTable),partNodes.dim))
        self.ptArea = np.zeros(shape=(len(partNodes.partNodesTable)))
        #Create coordVec array for later use in pddoW2
        i = 0
        for node_key_ID, node in partNodes.partNodesTable.items():
            self.nodeIdIndeces[i] = node_key_ID
            self.coordVec[i] = node.coords()
            self.ptArea[i] = node.vol()
            i += 1
            
        if self.hasInitialCrack == True:
            self.neighbors, self.start_idx, self.end_idx, self.n_neighbors = pddo.find_neighbors2(self.coordVec,1.01*self.delta,self.cracks)

        else:
            self.neighbors, self.start_idx, self.end_idx, self.n_neighbors = pddo.find_neighbors(self.coordVec,1.01*self.delta)
        print("the area:",self.ptArea)

        self.pd_point_count = self.n_neighbors.shape[0]
        self.pd_bond_count = self.neighbors.shape[0]
        self.bond_normals = pddo.calc_bond_normals(self.pd_point_count, self.pd_bond_count, self.coordVec, self.neighbors, self.start_idx, self.end_idx)
        self.curLiveBonds = np.ones_like(self.neighbors)
        self.curBondDamage = np.zeros_like(self.neighbors)
        self.init_BondLens = pddo.calc_bondLenghts(self.coordVec,self.neighbors,self.start_idx,self.end_idx)
        self.Gvec = pddo.gen_Gmat2D_fixed2(self.coordVec,self.neighbors,self.start_idx,self.end_idx,self.delta,self.ptArea)
        self.G11vec = self.Gvec[:,0]
        self.G12vec = self.Gvec[:,2]
        self.G22vec = self.Gvec[:,1]
