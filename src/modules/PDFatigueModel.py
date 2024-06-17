import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import numpy as np
from modules.NumericalModel import NumericalModel
from modules.data import HistoryOutput, State
import libs.pddopyW2 as pddo
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from numba import njit, jit, prange
from scipy.special import lambertw

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
        self.HistoryOutput = HistoryOutput()
        self.restartSim = True
    
    def gen_stiffness_matrix(self,LiveBonds: np.ndarray, bondDamage: np.ndarray) -> np.ndarray:
        """
        Generate the stiffness matrix for the given LiveBonds and bondDamage.

        Args:
            LiveBonds (np.ndarray): An array of live bonds.
            bondDamage (np.ndarray): An array of bond damage values.

        Returns:
            np.ndarray: The stiffness matrix.

        """
        stiffMat = np.zeros((self.FID.coordVec.shape[0]*2,self.FID.coordVec.shape[0]*2),dtype=float)
        return pddo._generate_stiffness_matrix2(self.FID.coordVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx, self.FID.G11vec, self.FID.G12vec, self.FID.G22vec, self.FID.muArr, LiveBonds, bondDamage, stiffMat)
    
    def gen_bond_stiffness_matrices(self) -> np.ndarray[float,2]:
        """
        Generate the bond stiffness matrices.

        Returns:
            np.ndarray[float,2]: The bond stiffness matrices.
        """
        return pddo._generate_bond_stiffnesses(self.FID.coordVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx, self.FID.G11vec, self.FID.G12vec, self.FID.G22vec, self.FID.muArr)

    def apply_displacement_BC(self,BCvec,stiffnessMat,RHSvec):
        """Does not work for compressed matrix forms
        Apply displacement boundary conditions to the given stiffness matrix and RHS vector.

        Parameters:
            BCvec (np.ndarray): The boundary condition vector.
            stiffnessMat (np.ndarray): The stiffness matrix.
            RHSvec (np.ndarray): The RHS vector.

        Returns:
            np.ndarray: The modified stiffness matrix and RHS vector with boundary conditions applied.
        """
        return pddo.applyDispBC2(BCvec,stiffnessMat,RHSvec,dim=self.FID.dim)

    def gen_bond_displacement_vecs(self,dispVec: np.ndarray[float,2]) -> np.ndarray:
        return pddo._generate_bond_displacement_vecs(dispVec,self.FID.neighbors, self.FID.start_idx, self.FID.end_idx)

    def calc_bond_stretches(self,cur_coordVec):
        """Calculate the stretches of each bond"""
        return (pddo.calc_bondLenghts(cur_coordVec,self.FID.neighbors,self.FID.start_idx,self.FID.end_idx)-self.FID.init_BondLens)/self.FID.init_BondLens

    def update_bond_damage(self,cur_bondStretches:np.ndarray, s1:np.ndarray[float,1], sc:np.ndarray[float,1])-> np.ndarray:
        return _calc_bond_damage(cur_bondStretches,s1,sc)
    
    def update_max_stretches_hist(self, current_state: State) -> np.ndarray[float, 1]:
        old_max_stretches = current_state.state_data["s_max_arr"]
        current_stretches = current_state.state_data["bondStretches"]
        new_max_stretches = _update_max_stretches_hist(old_max_stretches, current_stretches)
        return new_max_stretches


    def calc_static_damage_inc(self, state: State) -> np.ndarray[float, 1]:
        """
        Calculate the static damage increment based on the given state.
        This function calculates the static damage increment based on the s1_arr, sc_arr, s_max_arr, and bondStretches attributes of the FID object and the s_max_arr and stretch_arr attributes of the state object.
        It uses the _calc_static_damage_inc function to perform the calculation.

        Args:
            state (State): The current state of the system. Stores all data needed to progress to the next step. Currenlty, the user needs to make sure all the needed data is stored in teh State instance.

        Returns:
            np.ndarray[float,1]: The calculated static damage for each bond.

        """
        # if 0 in state.state_data["s_max_arr"]:
        #     raise ValueError("s_max = 0 -> division by zero. 0 encountered in static part of fatigue calculation.")
        
        if not state.has_state_data("s_max_arr"):
            raise ValueError("The state object does not have the required data to calculate the static damage increment -> 's_max_arr' is missing from the state object.")
        if not state.has_state_data("bondStretches"):
            raise ValueError("The state object does not have the required data to calculate the static damage increment -> 'stretch_arr' is missing from the state object.")
        
        s1_arr = self.FID.s0arr
        sc_arr = self.FID.scarr
        s_max_arr = state.state_data["s_max_arr"]
        stretch_arr = state.state_data["bondStretches"]
        live_bonds = state.state_data["LiveBonds"]
        delta_Ds_arr = _calc_static_damage_inc(live_bonds =live_bonds, s1_arr=s1_arr, sc_arr=sc_arr, s_max_arr=s_max_arr, sNpN_arr=stretch_arr)
        return delta_Ds_arr
    
    def calc_fatigue_damage_inc(self, state: State, cycle_increment: int):
        """
        Calculate the fatigue damage increment for each bond in the given state based on the specified cycle increment.

        Args:
            state (State): The current state of the system. Stores all data needed to progress to the next step. Currently, the user needs to make sure all the needed data is stored in the State instance.
            cycle_increment (int): The cycle increment to use for the fatigue damage calculation.

        Returns:
            np.ndarray[float, 1]: The calculated fatigue damage increment for each bond.

        Raises:
            ValueError: If the state object does not have the required data to calculate the fatigue damage increment. This can happen if the state object does not contain the "bondStretches", "s_max_arr", or "bondDamage" data.

        """
        # if 0 in state.state_data["s_max_arr"]:
        #     raise ValueError("s_max = 0 -> division by zero. 0 encountered in fatigue part of fatigue calculation.")

        delta_Df_arr = _calculate_fatigue_damage_increment(
        current_bond_stretches = state.state_data["bondStretches"],
        max_stretch_history = state.state_data["s_max_arr"],
        initial_damage = state.state_data["BondDamage"],
        live_bonds= state.state_data["LiveBonds"],
        s1_array = self.FID.s0arr,
        sc_array = self.FID.scarr,
        cycle_increment = cycle_increment,
        lambd = 0.5,
        mi = 0.7,
        C = 2e-6,
        beta = 2)

        return delta_Df_arr
    
    def calc_point_damage(self, state: State) -> np.ndarray:
        point_damage = np.zeros_like(self.FID.n_neighbors,dtype=float)
        bond_life = state.state_data["LiveBonds"]
        strt = self.FID.start_idx
        end = self.FID.end_idx
        for point in range(self.FID.coordVec.shape[0]): 
            point_damage[point] = 1.0 - (np.sum(bond_life[strt[point]:end[point]]))/np.float64(self.FID.n_neighbors[point])
        return point_damage
    
    def solve_lin_sys_for_f(self,disps) -> np.ndarray:
        """
        Solve the linear system for the force density vector (RHS).

        Args:
            disps (np.ndarray): The displacement vector.

        Returns:
            np.ndarray: The force density vector.
        """
        stiffMat = self.gen_stiffness_matrix(self.FID.curLiveBonds,self.FID.curBondDamage)
        forceDensVec = stiffMat @ disps
        return forceDensVec
    
    def solve_for_static_eq(self, staticSim: bool = True, fatigueSim: bool = False):
        """
        Solves for the equilibrium state of the system with the given "epsilon" as the maximum residual fraction.
        This function iteratively solves for the equilibrium state of the system using the Newton-Raphson method (calculate the stiffness matrix and take a step).
        Material nonlinearity is implemented as a reduction of the modulus of elasticity based on damage.
        This means that as the simulation iterates to get the equilibrium, it allways starts from the initial comfiguration (displacements do not get added together to get the final displacements).
        It starts with the initial live bonds and updates the stiffness matrix and bond damage based on the current live bonds.
        The displacement vector is calculated using sparse linear algebra operations, and the new coordinate vector is obtained by adding the displacement vector to the current coordinate vector.
        The bond stretches are calculated using the new coordinate vector, and the bond damage is updated accordingly.
        The live bonds are updated based on the current bond damage.
        The process is repeated until the maximum number of iterations is reached or the change in the residual force norm is smaller than the given epsilon.
        No geometric nonlinearity is implemented.
        
        Parameters:
            self (PDFatigueSolver): The instance of the PDFatigueSolver class.
        
        Returns:
            None
        
        Prints:
            Iteration number and the residual force norm at each iteration.
            The change in the residual force norm from the previous step
        """

        BCvec = self.FID.combined_BC_vec
        num_max_it = self.FID.num_max_it
        epsilon = self.FID.epsilon
        s0 = self.FID.s0arr
        sc = self.FID.scarr

        if num_max_it < 0 or type(num_max_it) != int:
            print("The maximum number of iterations can not be a negative value and it must be an integer type!")
        
        if epsilon <= 0:
            print("Epsilon can not be a negative value! Epsilon == 0 is not realistic and must be larger! (0 < epsilon)")
        if self.restartSim == True:
            self.FID.curLiveBonds = self.FID.initLiveBonds
            self.FID.curBondDamage = self.FID.initBondDamage


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
                # Equilibrium is reached for the change in stiffness. Now just update the displacements based on the updated stiffness matrix to get the final results
                _RHSvec = np.zeros(self.FID.coordVec.shape[0]*2)
                _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)
                _BC_stiffmatCSR = csr_matrix(_BC_stiffmat)
                _solu = spsolve(_BC_stiffmatCSR,_BC_RHSvec)
                # _solu, info = cg(_BC_stiffmatCSR,_BC_RHSvec)
                _disps = np.reshape(_solu,(int(_solu.shape[0]/2),2))
                
                # Save the data. Since we inside the eq. cond. statement the results that will be saved are from the equilibrium solution (to this is step)
                state = State()
                state.add_state_data("displacements", _disps)
                state.add_state_data("BondDamage", self.FID.curBondDamage)
                state.add_state_data("LiveBonds", self.FID.curLiveBonds)
                state.add_state_data("forceConvergence", self.FID.force_convergence)
                state.add_state_data("internalForces", _BC_RHSvec) 
                state.add_state_data("bondStretches", _cur_bond_stretches)
                state.is_from_step()
                self.HistoryOutput.add_state_to_history(state)
                print(f"Damage has converged in {iter} steps!")
                print(f"Residual forces/Externalforces = {_residual_force_norm}")
                print(f"Change of residual from previous step: {t1}")
                return 
            _residual_force_norm_old = _residual_force_norm
        self.result = _disps
        print("Solution did not converge!")
        return

   

    def solve_for_fatigue_eq(self, inputState: State, cycle_increment: int):
        ### Create local history output to be added to the globl histroy output after the simulation ends
        local_history_output = HistoryOutput()
        BCvec = self.FID.combined_BC_vec
        
        ### Check if inputState contains all required data
        if not inputState.has_state_data("s_max_arr"):
            inputState.add_state_data("s_max_arr", inputState.state_data["bondStretches"])
            if not inputState.has_state_data("bondStretches"):
                raise ValueError("WARNING: s_max not specified. Tries setting s_max array = bondStretches array but bondStretches array is not specified.")
            print("WARNING: s_max not specified. Setting s_max array = bondStretches array (bondStretches from inputState)")

        local_history_output.add_state_to_history(inputState)
        epsilon = self.FID.epsilon

        ## Check if inputState contains all required data
        if not inputState.has_state_data('BondDamage'):
            raise ValueError("Input state does not contain BondDamage!")
        
        if not inputState.has_state_data('LiveBonds'):
            raise ValueError("Input state does not contain LiveDamage!")
        
        newState = State()
        ## Initialize max stretches as the bond stretches for the initial iterstep
        # Take the data from the equilibrium and update the bond damage based on the fatigue part
        _bond_fatigue_damage_inc = self.calc_fatigue_damage_inc(state = inputState, cycle_increment = cycle_increment)
        # First we add the damage increment from fatigue to the absolute damage in each bond
        _newBondDamage = inputState.state_data["BondDamage"] + _bond_fatigue_damage_inc
        _newLiveBonds = _update_live_bonds(_newBondDamage)

        newState.add_state_data("bondStretches", inputState.state_data["bondStretches"])
        newState.add_state_data("bond_fatigue_damage_inc", _bond_fatigue_damage_inc)
        newState.add_state_data("bond_static_damage_inc", np.zeros_like(_bond_fatigue_damage_inc))
        newState.add_state_data("BondDamage", _newBondDamage)
        newState.add_state_data("LiveBonds", _newLiveBonds)
        initialLive_bonds = inputState.state_data["LiveBonds"]
        print("Max fatigue damage increment = ",_bond_fatigue_damage_inc.max())
        #These are placholders for old data that needs to beupdated inside the loop.
        newState.add_state_data("displacements", inputState.state_data["displacements"])
        newState.add_state_data("s_max_arr", inputState.state_data["s_max_arr"])
        
        _disps = inputState.state_data["displacements"] 
        _solu = _disps.flatten()#reshape(_disps.shape[0]*2, 1)
        
        # The stiffness matrix from the input state
        _RHSvec = np.zeros(self.FID.coordVec.shape[0]*2)
        _stiffmat = self.gen_stiffness_matrix(inputState.state_data["LiveBonds"],inputState.state_data["BondDamage"])
        _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)

        _residual_force_norm_old =  1
        max_iters=5
        for iter in range(max_iters):
            print("iter {}".format(iter))
            ### Inside the loop we now look for the equilibrium from equations 18 and 19 from the paper.

            ### We now have the damage state and can calculate for equilibrium
            ## Create a Stiffness matrix using updated bond damage states 
            _stiffmat_new = self.gen_stiffness_matrix(newState.state_data["LiveBonds"],newState.state_data["BondDamage"])
            
            ########### I need to check for eq. here! I already have u from the input state and a new K matrix
            
            ## Calculate internal forces using new stiffness matrix and old displacements -> (K_(iter+1) * u_(iter))
            _internal_force_vec = _stiffmat_new @ _solu
            ## Calculate residual forces ( |F_(iter) - K_(iter+1) * u_(iter)| / |F_(iter)| - 1)
            # Calculate difference in internal forces between the new and old stiffness matrix (the same disaplcements are used)
            _diff_in_internal_forces = np.linalg.norm(_BC_RHSvec-_internal_force_vec)
            self.test1 = _BC_RHSvec
            self.test2 = _internal_force_vec
            print("BCRHs",np.linalg.norm(_BC_RHSvec))
            print("INTFVEC",np.linalg.norm(_internal_force_vec))
            # Normalize the difference using the internal forces from the old stiffness matrix
            _diff_in_internal_forces_norm = (_diff_in_internal_forces/ np.linalg.norm(_BC_RHSvec))
            _residual_force_norm = np.abs(_diff_in_internal_forces_norm - 1)
            _chng_of_res_frce_from_prev_iter = np.abs((_residual_force_norm-_residual_force_norm_old)/_residual_force_norm_old)

            is_converged = 1-_chng_of_res_frce_from_prev_iter <= epsilon and _residual_force_norm <= epsilon 
            
            print("Normalized residual force = ", _residual_force_norm)
            print(f"Fraction of residual from previous iter: {_chng_of_res_frce_from_prev_iter}")
            print(f"Fractional change of residual from previous iter: {1-_chng_of_res_frce_from_prev_iter}")

            curState = State()
            curState.add_state_data("displacements", newState.state_data["displacements"]) #1 - For these displacements
            curState.add_state_data("bondStretches", newState.state_data["bondStretches"])  #2 - and these bond stretches (from the stretches above)
            curState.add_state_data("bond_fatigue_damage_inc", newState.state_data["bond_fatigue_damage_inc"]) #3 - we get this bond fatigue damage inc
            curState.add_state_data("bond_static_damage_inc", newState.state_data["bond_static_damage_inc"])    #4 - and this bond static damage inc.
            curState.add_state_data("BondDamage", newState.state_data["BondDamage"])    #5 - Resulting in this total damage
            curState.add_state_data("LiveBonds", newState.state_data["LiveBonds"])      #6 - and these live bonds.
            curState.add_state_data("s_max_arr", newState.state_data["s_max_arr"])      #7 - These are the same thorughout the loop. In a single cycle increment we dont update these!

            if is_converged:
                local_history_output.add_state_to_history(newState)

                self.local_history_output = local_history_output
                broken_bonds = initialLive_bonds.sum() - newState.state_data["LiveBonds"].sum()
                print("--------------------------------------------------------")
                print(f"Damage has converged in {iter} FATIGUE CRACK GROWTH iter/s!. Tolerance = {epsilon}")
                print(f"Residual forces/Externalforces = {_residual_force_norm}")
                print(f"Fraction of residual from previous iter: {_chng_of_res_frce_from_prev_iter}")
                print(f"Fractional change of residual from previous iter: {1-_chng_of_res_frce_from_prev_iter} -> Fraction + Fractional change = 1")
                print(f"Nubmer of bonds broken during fatigue step = {broken_bonds}.")
                return 
            elif iter == max_iters-1:
                local_history_output.add_state_to_history(newState)

                self.local_history_output = local_history_output
                broken_bonds = initialLive_bonds.sum() - newState.state_data["LiveBonds"].sum()
                print("--------------------------------------------------------")
                print(f"Damage has NOT converged in {iter} FATIGUE CRACK GROWTH iter/s!. Tolerance = {epsilon}")
                print(f"Residual forces/Externalforces = {_residual_force_norm}")
                print(f"Fraction of residual from previous iter: {_chng_of_res_frce_from_prev_iter}")
                print(f"Fractional change of residual from previous iter: {1-_chng_of_res_frce_from_prev_iter} -> Fraction + Fractional change = 1")
                print(f"Nubmer of bonds broken during fatigue step = {broken_bonds}.")
                return
            
            newState = State()
            _stiffmat = _stiffmat_new

            _RHSvec = np.zeros(self.FID.coordVec.shape[0]*2)
            _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)
            
            _BC_stiffmatCSR = csr_matrix(_BC_stiffmat)

            _solu_new = spsolve(_BC_stiffmatCSR,_BC_RHSvec)
            _disps_new = np.reshape(_solu_new,(int(_solu_new.shape[0]/2),2))
            
            _newCoordVec = self.FID.coordVec + _disps_new
            _new_bond_stretches = np.abs(self.calc_bond_stretches(_newCoordVec))
            
            newState.add_state_data("displacements", _disps_new)
            newState.add_state_data("bondStretches", _new_bond_stretches)
            newState.add_state_data("s_max_arr", curState.state_data["s_max_arr"]) # the same as before
            newState.add_state_data("LiveBonds", curState.state_data["LiveBonds"]) # just reuse the ones from before so that I skip the bonds that I know are broken

            _new_bond_static_damage_inc = self.calc_static_damage_inc(state = newState)
            _new_bond_fatigue_damage_inc = self.calc_fatigue_damage_inc(state = curState, cycle_increment = cycle_increment)
            
            newState.add_state_data("bond_fatigue_damage_inc", _new_bond_fatigue_damage_inc)
            newState.add_state_data("bond_static_damage_inc", _new_bond_static_damage_inc)

            _newBondDamage = inputState.state_data["BondDamage"] + _new_bond_static_damage_inc + _new_bond_fatigue_damage_inc
            _newLiveBonds = _update_live_bonds(_newBondDamage)

            newState.add_state_data("BondDamage", _newBondDamage)
            newState.add_state_data("LiveBonds", _newLiveBonds)

            _solu = _solu_new
        
        self.local_history_output = local_history_output
        self.restartSim = True
        # End of loop cycle -> go to new loop cycle



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
    live_bonds = np.ones_like(bond_damage)
    for bond in range(live_bonds.shape[0]):
        if bond_damage[bond] >= 1:
            live_bonds[bond] = 0
    return live_bonds

@njit
def _calc_static_damage_inc(
    live_bonds: np.ndarray[int,1],
    s1_arr: np.ndarray[float,1],
    sc_arr: np.ndarray[float,1],
    s_max_arr: np.ndarray[float,1],
    sNpN_arr: np.ndarray[float,1]
    ) -> np.ndarray[float,1]:
    """
    Calculate the incremental static damage based on the given parameters.

    Parameters:
        s1_arr (np.ndarray[float,1]): An array of s1 values. This array stores the stretch at which damage evolution starts for each bond. 
        sc_arr (np.ndarray[float,1]): An array of sc values. This array stores the stretch at which damage evolution ends based on the bi-linear model from eq. 8, for each bond. 
        s_max_arr (np.ndarray[float,1]): An array of s_max values. This array stores the maximum stretch for each bond in the history of the bond.
        sNpN_arr (np.ndarray[float,1]): An array of sNpN values. This array stores the stretch at the next fatigue increment. So basically the current stretch!

    Returns:
        delta_D_arr (np.ndarray[float,1]): An array of delta_D values. This array stores the incremental static damage for each bond.
    """

    delta_D_arr = np.zeros_like(s1_arr)
    for bond in range(s1_arr.shape[0]):
        s1 = s1_arr[bond]
        sc = sc_arr[bond]
        s_max = s_max_arr[bond]
        sNpN = sNpN_arr[bond]
        if  live_bonds[bond] ==True and sNpN >= sc and s_max != 0:
            delta_D = s1*sc/(sc-s1) * (1.0/s_max - 1.0/sNpN)
            delta_D_arr[bond] =  delta_D
    return delta_D_arr

@njit
def _update_max_stretches_hist(cur_max_stretches: np.ndarray[float, 1], current_stretches: np.ndarray[float,1]) -> np.ndarray[float, 1]:
    for bond in range(cur_max_stretches.shape[0]):
        if cur_max_stretches[bond] < current_stretches[bond]:
            cur_max_stretches[bond] = current_stretches[bond]

    return cur_max_stretches
@jit
def _calculate_fatigue_damage_increment(
    current_bond_stretches: np.ndarray[float,1],
    max_stretch_history: np.ndarray[float,1],
    initial_damage: np.ndarray[float,1],
    live_bonds: np.ndarray[int,1],
    s1_array: np.ndarray[float,1],
    sc_array: np.ndarray[float,1],
    cycle_increment: float,
    lambd: float,
    mi: float,
    C: float,
    beta: float
    ) -> np.ndarray:
    """
    Calculate the fatigue damage increment based on the given parameters.

    Parameters:
        current_bond_stretches (np.ndarray): An array of current bond stretches.
        max_stretch_history (np.ndarray): An array of maximum stretch history for each bond.
        initial_damage (np.ndarray): An array of initial damage for each bond.
        cycle_increment (float): The increment of the cycle.
        s1_array (np.ndarray): An array of s1 values.
        sc_array (np.ndarray): An array of sc values.
        lambd (float): The lambda value.
        mi (float): The mi value.
        C (float): The C value.
        beta (float): The beta value.

    Returns:
        fatigue_damage_increments (np.ndarray): An array of fatigue damage increments for each bond.
    """
    fatigue_damage_increments = np.zeros_like(current_bond_stretches)
    for bond in prange(current_bond_stretches.shape[0]):
        if current_bond_stretches[bond] >= max_stretch_history[bond] and live_bonds[bond] == 1 and current_bond_stretches[bond] != 0:
            a = -1 / (lambd * mi)
            b = -cycle_increment * ((lambd * mi * C) / (1 + beta))
            c = ((1 - mi) * max_stretch_history[bond] + mi * current_bond_stretches[bond]) / (sc_array[bond])
            c = c ** (1 + beta)
            d = lambd * initial_damage[bond]
            e = lambd * mi * (s1_array[bond] * sc_array[bond] / (sc_array[bond] - s1_array[bond])) * (1 / max_stretch_history[bond] - 1 / current_bond_stretches[bond])
            f = np.exp(d * e)
            fatigue_damage_increments[bond] = a * np.real(lambertw(b * c * f))
    return fatigue_damage_increments

