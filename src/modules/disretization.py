import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import modules.geometry as geometry
import numpy as np
import libs.pddopyW2 as pddo
from modules.data import get_user_decision

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
        self.initialCracks = np.zeros(shape=(0,2,2))

    def set_horizon(self, horizonRad: float) -> None:
        """
        Set the horizon radius of the object.

        Args:
            horizonRad (float): The radius of the horizon in units of your choice.

        Returns:
            None: This function does not return anything.
        """
        self.delta = float(horizonRad)

    def create_crack(self,p1x,p1y,p2x,p2y) -> None:
        """
        Creates a crack and adds it to the initial cracks list.
        The bonds that intersect with any of the bonds in this list is not added to the family of a material point.
        Therefore, when creating A matrices (from PDDO to create g functions), the points connected by these bonds are not included in the summation.

        Parameters:
            p1x (float): The x-coordinate of the first point of the crack.
            p1y (float): The y-coordinate of the first point of the crack.
            p2x (float): The x-coordinate of the second point of the crack.
            p2y (float): The y-coordinate of the second point of the crack.

        Returns:
            None
        """
        crack = np.array([[[p1x,p1y],[p2x,p2y]]],dtype=float)
        self.initialCracks = np.append(self.initialCracks,crack,axis=0)
        self.hasInitialCrack = True
        self.define_crack_mode("familyExclusion")

    def deactivate_cracks(self) -> None:
        """
        Deactivates the cracks by setting the `hasInitialCrack` attribute to False.

        Returns:
            None: This function does not return anything.
        """
        self.hasInitialCrack = False

    def activate_cracks(self) -> None:
        """
        Activates the cracks by setting the `hasInitialCrack` attribute to True.

        This function does not take any parameters.

        Returns:
            None: This function does not return anything.
        """
        self.hasInitialCrack = True

    def is_crack_active(self) -> bool:
        """
        Returns a boolean value indicating whether the crack is currently active or not.

        Returns:
            bool: A boolean value indicating whether the crack is currently active or not.
        """
        return self.hasInitialCrack
    
    def define_crack_mode(self, crackMode: str = "familyExclusion") -> None:
        """
        crackMode = "familyExclusion" means that the crack should be used to exclude nodes from the family of a material point.
        crackMode = "bondIntersection" means that the crack should not be used to exclude nodes from the family of a material point, but rather to create an additional live bonds array.
        """
        self.initial_crack_mode = crackMode

    def get_node_family_IDs(self,nodeID: int) -> np.ndarray[int,1]:
        """
        Returns ID's of the nodes that are inside the family of the given node ID.

        Args:
            nodeID (int): The ID of the node for which the family IDs are to be retrieved.

        Returns:
            np.ndarray[int,1]: An array of node family IDs that are included in the node defined with `nodeID`.

        This function finds the position of the given node ID in the `nodeIdIndeces` array. It then uses the position to
        find the corresponding indices in the `start_idx` and `end_idx` arrays. It slices the `neighbors` array using
        these indices and retrieves the node family IDs from the `nodeIdIndeces` array. Finally, it returns the node family
        IDs as a 1-dimensional numpy array.
        """
        # nodeIdPosition = np.where(self.nodeIdIndeces == nodeID)[0]
        firstMemberIndex = self.start_idx[nodeID]
        lastMemberIndex = self.end_idx[nodeID]
        nodeFamily = self.neighbors[firstMemberIndex:lastMemberIndex]
        return nodeFamily

    def get_node_family_coords(self,nodeID: int) -> np.ndarray[int,2]:
        """
        Get the coordinates of the nodes that are part of the family of a given node ID.

        Parameters:
            nodeID (int): The ID of the node whose family members coordinates are to be retireved.

        Returns:
            np.ndarray[int,2]: An array of shape (n, 2) containing the coordinates of the nodes in the family of the given "nodeID" parameter.
        """
        firstMemberIndex = self.start_idx[nodeID]
        lastMemberIndex = self.end_idx[nodeID]
        nodeFamilyCoords = self.coordVec[self.neighbors[firstMemberIndex:lastMemberIndex]]
        return nodeFamilyCoords


    def generate_bonds(self,partNodes: geometry._PartNodes) -> None:
        """
        Generates bonds between nodes in a given geometry.

        Parameters:
            partNodes (geometry._PartNodes): The PartNodes object containing the nodes for which bonds need to be generated.

        Returns:
            None: This function does not return anything.

        This function generates bonds between nodes in a given geometry. It takes a PartNodes object as input, which contains the nodes for which bonds need to be generated.
        The function first checks if all the necessary data for generating bonds is available.
        Then, it saves the IDs and coordinates of each node in the partNodesTable to separate arrays.
        It creates a coordVec array for later use in pddoW2. 
        If the object has an initial crack, it calls the find_neighbors2 function from the pddo module with the coordVec, a scaled delta value, and the initial cracks as parameters.
        Otherwise, it calls the find_neighbors function from the pddo module with the coordVec and a scaled delta value (this function does not tak into account the initial cracks).
        The function then calculates the number of points and bonds in the generated neighbors.
        It calculates the bond normals using the calc_bond_normals function from the pddo module.
        It initializes the live bonds and bond damage arrays with ones and zeros respectively.
        It calculates the initial bond lengths using the calc_bondLenghts function from the pddo module.
        It generates the Gvec matrix using the gen_Gmat2D_fixed2 function from the pddo module.
        Finally, it assigns the values from the Gvec matrix to the G11vec, G12vec, and G22vec arrays.
        """
        #Check if everything needed for this function is satisfied!
        if partNodes.partNodesTable == {}:
            raise ValueError('The partNodesTable is empty. The nodes are not generated yet.')

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
            
        if self.hasInitialCrack == True and self.initial_crack_mode == "familyExclusion":
            self.neighbors, self.start_idx, self.end_idx, self.n_neighbors = pddo.find_neighbors2(self.coordVec,1.01*self.delta,self.initialCracks)
            self.initLiveBonds = np.ones_like(self.neighbors)
            self.curLiveBonds = np.ones_like(self.neighbors)

        elif self.hasInitialCrack == True and self.initial_crack_mode == "bondIntersection":
            print("Itersect style")
            self.neighbors, self.start_idx, self.end_idx, self.n_neighbors, self.initLiveBonds = pddo.find_neighbors3(self.coordVec,1.01*self.delta,self.initialCracks)
            self.curLiveBonds = self.initLiveBonds

        else:
            print("Undef")
            self.neighbors, self.start_idx, self.end_idx, self.n_neighbors = pddo.find_neighbors(self.coordVec,1.01*self.delta)
            self.initLiveBonds = np.ones_like(self.neighbors)
            self.curLiveBonds = np.ones_like(self.neighbors)

        self.pd_point_count = self.n_neighbors.shape[0]
        self.pd_bond_count = self.neighbors.shape[0]
        self.bond_normals = pddo.calc_bond_normals(self.pd_point_count, self.pd_bond_count, self.coordVec, self.neighbors, self.start_idx, self.end_idx)
        self.initBondDamage = np.zeros_like(self.neighbors)
        self.curBondDamage = np.zeros_like(self.neighbors)
        self.init_BondLens = pddo.calc_bondLenghts(self.coordVec,self.neighbors,self.start_idx,self.end_idx)
        self.Gvec = pddo.gen_Gmat2D_fixed2(self.coordVec,self.neighbors,self.start_idx,self.end_idx,self.delta,self.ptArea)
        self.G11vec = self.Gvec[:,0]
        self.G12vec = self.Gvec[:,2]
        self.G22vec = self.Gvec[:,1]
