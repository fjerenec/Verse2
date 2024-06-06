class PDVerseSolver:
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
        BCvec = self.FID.combined_BC_vec
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