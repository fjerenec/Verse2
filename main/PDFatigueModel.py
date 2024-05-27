import numpy as np
from NumericalModel import NumericalModel
import pddopyW2 as pddo
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from numba import njit

class FatigueInputData:
    def __init__(self, numModel: NumericalModel) -> None:
        self.coordVec = numModel.Geometry.part_nodes.coordVec
        self.selectedDiscretization = numModel.Discretizations.selectedDiscretization
        self.delta = self.selectedDiscretization.delta
        self.ptVolumes = self.selectedDiscretization.ptArea
        self.initialCracks = self.selectedDiscretization.initialCracks
        self.neighbors   = self.selectedDiscretization.neighbors
        self.start_idx   = self.selectedDiscretization.start_idx
        self.end_idx     = self.selectedDiscretization.end_idx
        self.n_neighbors = self.selectedDiscretization.n_neighbors

        self.pd_point_count = self.selectedDiscretization.pd_point_count
        self.pd_bond_count  = self.selectedDiscretization.pd_bond_count
        self.bond_normals   = self.selectedDiscretization.bond_normals
        self.initLiveBonds  = self.selectedDiscretization.initLiveBonds
        self.initBondDamage = self.selectedDiscretization.initBondDamage
        self.curLiveBonds   = self.selectedDiscretization.curLiveBonds
        self.curBondDamage  = self.selectedDiscretization.curBondDamage
        self.init_BondLens  = self.selectedDiscretization.init_BondLens
        self.Gvec   = self.selectedDiscretization.Gvec
        self.G11vec = self.selectedDiscretization.G11vec
        self.G12vec = self.selectedDiscretization.G12vec
        self.G22vec = self.selectedDiscretization.G22vec

        self.materialsByIDTable = numModel.Materials.materialsByIDTable
        self.materialSectionsTable = numModel.MaterialSections.materialSectionsTable
        self.materialInterfacesTable = numModel.MaterialInterfaces.materialInterfacesTable
        self.materialInterfaceFound = bool(self.materialInterfacesTable) 
        self.gen_bond_material_array()
        self.dispLoadsTable = numModel.Loads.dispLoadsTable
        self.forceLoadsTable = numModel.Loads.forceLoadsTable
        self.create_combined_disp_BC_vec()
        
        self.num_max_it = 10
        self.epsilon = 0.1
        self.gen_damage_evolution_stretch_arrays()
        self.gen_bond_emod_array()
        self.gen_bond_mu_array()
        self.mu = 1
        self.dim = 2
        
        self.force_convergence = []

    def gen_bond_material_array(self):
        self.bondMaterialIDarray = np.empty_like(self.neighbors)
        # Loop over all bonds and extract the IDs of the two points in the bond
        for point_id_1 in range(self.coordVec.shape[0]):
            for bond in range(self.start_idx[point_id_1], self.end_idx[point_id_1]):
                point_id_2 = self.neighbors[bond]

                if self.materialInterfaceFound:
                    # Loop over all MaterialInterfaces
                    for interface_name, material_interface in self.materialInterfacesTable.items():
                        # Check if the points are in different MaterialSections of this MaterialInterface
                        point_1_in_section_1 = point_id_1 in material_interface.materialSection1.nodeSet.IDTable.keys()
                        point_1_in_section_2 = point_id_1 in material_interface.materialSection2.nodeSet.IDTable.keys()
                        point_2_in_section_1 = point_id_2 in material_interface.materialSection1.nodeSet.IDTable.keys()
                        point_2_in_section_2 = point_id_2 in material_interface.materialSection2.nodeSet.IDTable.keys()
                        
                        # If points are in different sections of this interface, retrieve Material id
                        isInMatInterface = (point_1_in_section_1 and point_2_in_section_2) or (point_1_in_section_2 and point_2_in_section_1)
                        if isInMatInterface:
                            self.bondMaterialIDarray[bond] = material_interface.material.materialID
                            # Store relevant data from Material object to a new array
                            # Append to the new array, or perform any other required action
                            break

                    if not isInMatInterface:
                        for section_name, material_section in self.materialSectionsTable.items():
                        #Loop over the MaterialSections -> This means the bond was not in any interface
                        #If the bond is not in a material interface, then both point MUST be in the same material section
                        #So i only need to check one point!
                            point_in_material_section = point_id_1 in material_section.nodeSet.IDTable.keys()
                            if point_in_material_section:
                                self.bondMaterialIDarray[bond] = material_section.material.materialID
                                break
                #If there is no Material Interface that means that the whole model is made up from only one material.
                #This also means that there is only one material section. So i can simply create an array filled with the material section material id
                else:
                    #Check there is only one material interface:
                    if len(self.materialSectionsTable) != 1:
                        return print("There is more than one material section and no material interface in the model. A material interface must be defined in the case of multiple material section!")
                    for material_section_same, material_section in self.materialSectionsTable.items():
                        #For loop not needed since in this case there is only one material section.
                        self.bondMaterialIDarray[:] = material_section.material.materialID
    
    def gen_damage_evolution_stretch_arrays(self):
        self.s0arr = np.zeros_like(self.bondMaterialIDarray, dtype = float)
        self.scarr = np.zeros_like(self.bondMaterialIDarray, dtype = float)
        for i,materialID in enumerate(self.bondMaterialIDarray):
            self.s0arr[i] = self.materialsByIDTable[materialID].s0
            self.scarr[i] = self.materialsByIDTable[materialID].sc

    def gen_bond_emod_array(self):
        self.emodArr = np.empty_like(self.bondMaterialIDarray,dtype = float)
        for i, materialID in enumerate(self.bondMaterialIDarray):
            self.emodArr[i] = self.materialsByIDTable[materialID].Emod

    def gen_bond_mu_array(self):
        if self.emodArr is None:
            self.gen_bond_emod_array()

        self.muArr = np.empty_like(self.emodArr,dtype = float)
        for i, emod in enumerate(self.emodArr):
            self.muArr[i] = emod/(2*(float(1+0.25)))

    def create_combined_disp_BC_vec(self):
        numOfPtsInAllSets = 0
        for name, displacement_load in self.dispLoadsTable.items():
            numOfPtsInDispLoad = displacement_load.nodeSet.get_number_of_points()
            numOfPtsInAllSets += numOfPtsInDispLoad

        self.combined_BC_vec = np.ndarray(shape=(numOfPtsInAllSets,7))
        previousSetsSum = 0
        currentSetLength = 0
        for name, displacement_load in self.dispLoadsTable.items():
            currentSetLength = displacement_load.nodeSet.get_number_of_points()
            currentSetsSum = previousSetsSum+currentSetLength
            self.combined_BC_vec[previousSetsSum : currentSetsSum] = displacement_load.BC_vec
            previousSetsSum = currentSetsSum

class PDFatigueSolver:
    def __init__(self, numModel: NumericalModel) -> None:
        """FID = FatigueInputData"""
        self.FID = FatigueInputData(numModel)
        pass
    
    def gen_stiffness_matrix(self,LiveBonds: np.ndarray, bondDamage: np.ndarray) -> np.ndarray:
        stiffMat = np.zeros((self.FID.coordVec.shape[0]*2,self.FID.coordVec.shape[0]*2),dtype=float)
        return pddo._generate_stiffness_matrix2(self.FID.coordVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx, self.FID.G11vec, self.FID.G12vec, self.FID.G22vec, self.FID.muArr, LiveBonds, bondDamage, stiffMat)
    
    def gen_bond_stiffness_matrices(self) -> np.ndarray:
        return pddo._generate_bond_stiffnesses(self.FID.coordVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx, self.FID.G11vec, self.FID.G12vec, self.FID.G22vec, self.FID.muArr)

    def apply_displacement_BC(self,BCvec,stiffnessMat,RHSvec):
        """Only works for dense matrix form"""
        return pddo.applyDispBC2(BCvec,stiffnessMat,RHSvec,dim=self.FID.dim)

    def gen_bond_displacement_vecs(self,dispVec: np.ndarray[float,2]) -> np.ndarray:
        return pddo._generate_bond_displacement_vecs(dispVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx)

    def calc_bond_stretches(self,cur_coordVec):
        """Calculate the stretches of each bond"""
        return (pddo.calc_bondLenghts(cur_coordVec,self.FID.neighbors,self.FID.start_idx,self.FID.end_idx)-self.FID.init_BondLens)/self.FID.init_BondLens

    def update_bond_damage(self,cur_bondStretches:np.ndarray, s1:np.ndarray[float,1], sc:np.ndarray[float,1])-> np.ndarray:
        return _calc_bond_damage(cur_bondStretches,s1,sc)
    
    def solve_lin_sys_for_f(self,disps) -> np.ndarray:
        stiffMat = self.gen_stiffness_matrix(self.FID.curLiveBonds,self.FID.curBondDamage)
        forceDensVec = stiffMat @ disps
        return forceDensVec
    
    def solve_for_eq3(self):
        """Solves for equlibirum state of the system with desegnated "epsilon" as the maximum residual fraction"""
        BCvec = self.FID.combined_BC_vec#[:-1]
        num_max_it = self.FID.num_max_it
        epsilon = self.FID.epsilon
        s0 = self.FID.s0arr
        sc = self.FID.scarr

        if num_max_it < 0 or type(num_max_it) != int:
            print("The maximum number of iterations can not be a negative value and it must be an integer type!")
        
        if epsilon <= 0:
            print("Epsilon can not be a negative value! Epsilon == 0 is not realistic and must be larger! (0 < epsilon)")

        self.FID.curLiveBonds = self.FID.initLiveBonds
        _stiffmat = self.gen_stiffness_matrix(self.FID.curLiveBonds, self.FID.curBondDamage)
        _residual_force_norm_old =  1
        for iter in range(num_max_it):# and error > epsilon:
            print("Iteration {}".format(iter))
            _RHSvec = np.zeros(self.FID.coordVec.shape[0]*2)
            _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)
            _BC_stiffmatCSR = csr_matrix(_BC_stiffmat)
            _solu = spsolve(_BC_stiffmatCSR,_BC_RHSvec)
            # _solu, info = cg(_BC_stiffmatCSR,_BC_RHSvec)
            _disps = np.reshape(_solu,(int(_solu.shape[0]/2),2))
            _newCoordVec = self.FID.coordVec + _disps
            _cur_bond_stretches = np.abs(self.calc_bond_stretches(_newCoordVec))
            self.FID.curBondDamage = self.update_bond_damage(_cur_bond_stretches,s0,sc)
            self.FID.curLiveBonds = _update_live_bonds(self.FID.curBondDamage)
            _stiffmat = self.gen_stiffness_matrix(self.FID.curLiveBonds, self.FID.curBondDamage)
            _internal_force_vec = _stiffmat @ _solu
            _residual_force_norm = np.abs(np.linalg.norm(_BC_RHSvec-_internal_force_vec) / np.linalg.norm(_BC_RHSvec)-1)
            print("Residual force norm = ",_residual_force_norm)
            t1= np.abs(1-_residual_force_norm/_residual_force_norm_old)
            self.FID.force_convergence.append(_residual_force_norm)
            print(f"Change of residual from previous step: {t1}")
            if _cur_bond_stretches.max() <= s0.min():
                print("Applied load was not large enough to cause damage!")
                break
            if t1<= epsilon:
                print(f"Damage has converged in {iter} steps! Residual forces/Externalforces = {_residual_force_norm}.Change of residual from previous step: {t1}")
                print(f"Residual forces/Externalforces = {_residual_force_norm}")
                print(f"Change of residual from previous step: {t1}")
                self.result = _disps
                return 
            _residual_force_norm_old = _residual_force_norm
        print("Solution did not converge!")
        self.result = _disps
        return

def _calc_bond_damage(cur_bondStretches:np.ndarray, s0arr:np.ndarray[float,1], scarr:np.ndarray[float,1]) -> np.ndarray:
    new_damage = np.zeros(cur_bondStretches.shape[0])
    for bond in range(cur_bondStretches.shape[0]):
        s0 = s0arr[bond]
        sc = scarr[bond]
        if  s0 <= cur_bondStretches[bond] <= sc:
            new_damage[bond] = sc/(sc-s0)*(1- s0/cur_bondStretches[bond])
        if cur_bondStretches[bond] >= sc:
            new_damage[bond] = 1
    return new_damage

@njit
def _update_live_bonds(bond_damage: np.ndarray[float,1]) -> np.ndarray[int,1]:
    live_bonds = np.zeros_like(bond_damage)
    for bond in range(live_bonds.shape[0]):
        if bond_damage[bond] < 1:
            live_bonds[bond] = 1
    return live_bonds

