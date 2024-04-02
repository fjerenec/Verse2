import numpy as np
from NumericalModel import NumericalModel
import pddopyW2 as pddo
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

class FatigueInputData:
        def __init__(self, numModel: NumericalModel) -> None:
            self.coordVec = numModel.Geometry.part_nodes.coordVec
            self.selectedDiscretisation = numModel.Discretisations.selectedDiscretisation
            self.delta = self.selectedDiscretisation.delta
            self.initialCracks = self.selectedDiscretisation.initialCracks
            self.neighbors   = self.selectedDiscretisation.neighbors
            self.start_idx   = self.selectedDiscretisation.start_idx
            self.end_idx     = self.selectedDiscretisation.end_idx
            self.n_neighbors = self.selectedDiscretisation.n_neighbors

            self.pd_point_count = self.selectedDiscretisation.pd_point_count
            self.pd_bond_count  = self.selectedDiscretisation.pd_bond_count
            self.bond_normals   = self.selectedDiscretisation.bond_normals
            self.curLiveBonds   = self.selectedDiscretisation.curLiveBonds
            self.curBondDamage  = self.selectedDiscretisation.curBondDamage
            self.init_BondLens  = self.selectedDiscretisation.init_BondLens
            self.Gvec   = self.selectedDiscretisation.Gvec
            self.G11vec = self.selectedDiscretisation.G11vec
            self.G12vec = self.selectedDiscretisation.G12vec
            self.G22vec = self.selectedDiscretisation.G22vec

            # self.emodArray = 
            # self.muArray = 

            self.force_convergence = []

class PDFatigueSolver:
    def __init__(self, numModel: NumericalModel) -> None:
        """FID = FatigueInputData"""
        self.FID = FatigueInputData(NumericalModel)
        pass
    
    def gen_stiffness_matrix(self,LiveBonds: np.ndarray, bondDamage: np.ndarray) -> np.ndarray:
        stiffMat = np.zeros((self.FID.coordVec.shape[0]*2,self.FID.coordVec.shape[0]*2),dtype=float)
        return pddo._generate_stiffness_matrix(self.FID.coordVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx, self.FID.G11vec, self.FID.G12vec, self.FID.G22vec, self.material.mu, LiveBonds, bondDamage, stiffMat)
    
    def apply_displacement_BC(self,BCvec,stiffnessMat,RHSvec):
        """Only works for dense matrix form"""
        return pddo.applyDispBC(BCvec,stiffnessMat,RHSvec)
    
    def calc_bond_stretches(self,cur_coordVec):
        """Calculate the stretches of each bond"""
        return (pddo.calc_bondLenghts(cur_coordVec,self.FID.neighbors,self.FID.start_idx,self.FID.end_idx)-self.FID.init_BondLens)/self.FID.init_BondLens

    def update_bond_damage(self,cur_bondStretches:np.ndarray, s1:float, sc:float)-> np.ndarray:
        return _calc_bond_damage(cur_bondStretches,s1,sc)
    
    def solve_for_eq3(self, BCvec:np.ndarray, s1:float, sc:float , num_max_it:int, epsilon:float) -> np.ndarray:
        """Solves for equlibirum state of the system with desegnated "epsilon" as the maximum residual fraction"""

        if num_max_it < 0 or type(num_max_it) != int:
            print("The maximum number of iterations can not be a negative value and it must be an integer type!")
        
        if epsilon <= 0:
            print("Epsilon can not be a negative value! Epsilon == 0 is not realistic and must be larger! (0 < epsilon)")

        _stiffmat = self.gen_stiffness_matrix(self.FID.curLiveBonds, self.FID.curBondDamage)
        _residual_force_norm_old =  1
        for iter in range(num_max_it):# and error > epsilon:
            print("Iteration {}".format(iter))
            _RHSvec = np.zeros(self.FID.coordVec.shape[0]*2)
            _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)
            _BC_stiffmatCSR = csr_matrix(_BC_stiffmat)
            _solu = spsolve(_BC_stiffmatCSR,_BC_RHSvec)
            _disps = np.reshape(_solu,(int(_solu.shape[0]/2),2))
            _newCoordVec = self.FID.coordVec + _disps
            _cur_bond_stretches = np.abs(self.calc_bond_stretches(_newCoordVec))
            self.FID.curBondDamage = self.update_bond_damage(_cur_bond_stretches,s1,sc)
            # #need a line to update "LiveBonds" array!
            _stiffmat = self.gen_stiffness_matrix(self.FID.curLiveBonds, self.FID.curBondDamage)
            _internal_force_vec = _stiffmat @ _solu
            _residual_force_norm = np.abs(np.linalg.norm(_BC_RHSvec-_internal_force_vec) / np.linalg.norm(_BC_RHSvec)-1)
            print("Residual force norm = ",_residual_force_norm)
            t1= np.abs(1-_residual_force_norm/_residual_force_norm_old)
            self.FID.force_convergence.append(_residual_force_norm)
            print(f"Change of residual from previous step: {t1}")
            if _cur_bond_stretches.max() <= s1:
                print("Applied load was not large enough to cause damage!")
                break
            if t1<= epsilon:
                print(f"Damage has converged in {iter} steps! Residual forces/Externalforces = {_residual_force_norm}.Change of residual from previous step: {t1}")
                print(f"Residual forces/Externalforces = {_residual_force_norm}")
                print(f"Change of residual from previous step: {t1}")
                return _disps
            _residual_force_norm_old = _residual_force_norm
        print("Solution did not converge!")
        return _disps

def _calc_bond_damage(cur_bondStretches:np.ndarray, s1:float, sc:float) -> np.ndarray:
    new_damage = np.zeros(cur_bondStretches.shape[0])
    for bond in range(cur_bondStretches.shape[0]):
        if  s1 <= cur_bondStretches[bond] <= sc:
            new_damage[bond] = sc/(sc-s1)*(1- s1/cur_bondStretches[bond])
        if cur_bondStretches[bond] >= sc:
            new_damage[bond] = 1
    return new_damage