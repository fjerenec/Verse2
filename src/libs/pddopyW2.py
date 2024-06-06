from numba import njit, prange
import numpy as np
from scipy.special import factorial
from scipy.sparse import csr_matrix
# from pypardiso import spsolve
from scipy.sparse.linalg import spsolve #Change this back to pypardiso! Using scipy only for MacOD
from scipy.sparse.linalg import cg

@njit
def find_neighbors(coords, r):
    """
    Find the neigbors of point i that are in range of delta. The funtions works on multidimensional points.
    Inputs:
    -----------
    coords : array of array of floats (coordinates of points=. shape = [number of points, dimensionality of points]. Coordinates should be floats.
    r      : radius of includivity. Points j that are distanced r or less, from point i, are considered to be members of point i. Should be float

    Outputs:
    -----------
    neighbors   : 1D array of floats (ID numbers of points). Contains the ID's of the neighbors of all points starting with the ID's of neighbors of point 0.
    start_index : 1D array of ints. Component "i" of the start_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" start.
    end_index   : 1D array of ints. Component "i" of the end_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" end.
    n_neighbors : 1D array of ints. Component "i" contains the number of neighbors that the point with ID = "i" has.

    """
    n_points = coords.shape[0]
    max_neighbors = 120 # This should change based on the dimensionality of the points. 28 is just for 2d and delta = 3 * dx.
    neighbors = np.full((n_points*max_neighbors), -1, dtype=np.int64)
    start_index = np.zeros(n_points, dtype=np.int64)
    end_index = np.zeros(n_points, dtype=np.int64)
    n_neighbors = np.zeros(n_points, dtype=np.int64)
    current_neighbor = 0
    for i in range(n_points):
        for j in range(n_points):
            if i!=j:
                dist = 0
                for m in range(coords.shape[1]):
                    dist += (coords[i,m]-coords[j,m])**2
                dist = dist**0.5
                if dist <= r:
                    neighbors[current_neighbor] = j
                    n_neighbors[i] += 1
                    current_neighbor += 1
    end_index = np.cumsum(n_neighbors)
    start_index[1:] = end_index[:n_points-1]
    return neighbors[:end_index[n_points-1]], start_index,end_index, n_neighbors

@njit
def find_neighbors2(coords, r, cracks):
    """
    Find the neigbors of point i that are in range of delta. The funtions works on multidimensional points. The same as find_neighbors2 but allows for cracks.
    Inputs:
    -----------
    coords : array of array of floats (coordinates of points=. shape = [number of points, dimensionality of points]. Coordinates should be floats.
    r      : radius of includivity. Points j that are distanced r or less, from point i, are considered to be members of point i. Should be float
    cracks : 2d array of floats. Contains the couples of points that define all the cracks inthe model. shape = [number of crack lines,2]
                example: cracks = [[[x0,y0],[x1,y1]],
                                   [[x2,y2],[x3,y3]]]  -> [x0,y0] is the first point of the crack 0th crack, and [x1,y1] is the second point
    Outputs:
    -----------
    neighbors   : 1D array of floats (ID numbers of points). Contains the ID's of the neighbors of all points starting with the ID's of neighbors of point 0.
    start_index : 1D array of ints. Component "i" of the start_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" start.
    end_index   : 1D array of ints. Component "i" of the end_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" end.
    n_neighbors : 1D array of ints. Component "i" contains the number of neighbors that the point with ID = "i" has.

    """
    n_points = coords.shape[0]
    max_neighbors = 120 # This should change based on the dimensionality of the points. 28 is just for 2d and delta = 3 * dx.
    neighbors = np.full((n_points*max_neighbors), -1, dtype=np.int64)
    start_index = np.zeros(n_points, dtype=np.int64)
    end_index = np.zeros(n_points, dtype=np.int64)
    n_neighbors = np.zeros(n_points, dtype=np.int64)
    current_neighbor = 0
    for i in range(n_points):
        for j in range(n_points):
            if i!=j:
                dist = 0
                for m in range(coords.shape[1]):
                    dist += (coords[i,m]-coords[j,m])**2
                dist = dist**0.5
                if dist <= r:
                    alive = True
                    for curcrack in range(cracks.shape[0]):
                        #if intersect2(point1 of crack,point2 of crack,point1 of bond,point2 of crack)
                        alive = alive*np.logical_not(intersect2(cracks[curcrack,0],cracks[curcrack,1],coords[i],coords[j])) #first the cracks then the point
                    if alive == True:
                        neighbors[current_neighbor] = j
                        n_neighbors[i] += 1
                        current_neighbor += 1
    end_index = np.cumsum(n_neighbors)
    start_index[1:] = end_index[:n_points-1]
    return neighbors[:end_index[n_points-1]], start_index,end_index, n_neighbors

@njit
def ccw2(A,B,C):
	#return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
	return ((C[1]-A[1])*(B[0]-A[0]))>((B[1]-A[1])*(C[0]-A[0]))
@njit
def intersect2(A,B,C,D):
	#return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
	return (ccw2(A,C,D) != ccw2(B,C,D)) and (ccw2(A,B,C) != ccw2(A,B,D))

def tse_terms(M,N):
    """
    Calculate the number of TSE terms based on the dimensionality of PD points "M" and the specified maximum order of TSE terms "N"
    M must! be higher or equal to N. The output includes the term x1^0 * x2^0 * ... * xi^0. For a relative function PDDO the output need to be
    lowered by 1.
    Inputs:
    ---------
    M : int. Dimensionality of pd points
    N : int. Maximum order of TSE terms

    Output:
    ---------
    D : int. Number of tse terms.
    """
    D = np.int32(0)
    m = 1
    n=N
    def loop1(m,n):
        nonlocal D
        if (m > 0 and m <= M):
            for i in range(0,n+1):
                loop1(m+1,n-i)
                if m == M:
                    D = D + 1
        return D
    return loop1(m,n)

def tse_pows(N,M,D):
    """
    Compute the powers of each xsi of each component of xsi (each dimension of xsi) and for each TSE term.
    Xsi represents a bond. Another way to look at a bond is simply as a connection between two points in a family.
    Since xsi is a connection it has some length in every dimension so it is a vector of relative position between two points
    with vector components xsi = (xsi1,xsi2,xsi3,..,xsiM)! -> could also be wirtten as xsi = (x,y,z,..,t), but this way you quickly
    run out of characters to represent dimensions.

    Inputs:
    ---------
    M : int. Dimensionality of pd points
    N : int. Maximum order of TSE terms
    D : int. Number of tse terms.

    Output:
    ---------
    p : 2d array[:,:]. Contains the powers of each xsi of each component of xsi (each dimension of xsi) and for each TSE term.
                        p.shape[0] = "number of tse terms".
                        p.shape[1] = "M" (the dimensionllality of the points).
                            Each column in "p" (p[:,m]) contains the powers for one component of xsi.
    """

    icount = 0 #A counter that keeps track of the current number of bonds, while looping through each points and finding their family membrs.
    p = np.zeros((D,M),dtype=np.int64)
    num1=np.zeros((10,1),dtype=np.int64) # Temporary parameter
    num = np.zeros((10,1),dtype=np.int64) # Temporary parameter
    m=np.int64(1)
    n=N

    def loop2(m,n,num):
        nonlocal icount
        nonlocal p
        nonlocal M
        num1=num
        if m>0 and m<=M:
            for i in range(n+1):
                num1[m-1]=i
                loop2(m+1,n-i,num1)
                if m==M:
                    icount = icount+1
                    for m in range(1,M+1):
                        p[icount-1,m-1]=num1[m-1]


        return p
    return loop2(m,n,num)

@njit
def axis1prod(array):
    result = array[:, 0]
    for i in range(1, array.shape[1]):
        result *= array[:, i]
    return result


def axis1fact(array):
    return axis1prod(factorial(array)).reshape(array.shape[0],1)

@njit
def matinv(A):
    return np.linalg.inv(A)

def gen_bmat(porders,pows):
    """
    Function to compute the right hand side vector (or matrix if you want to compute more than 1 derivative). Relative function!
    Inputs
    -------
    porders: 2d array of floats. Contains the orders of differentiation for each dimension.
                porders.shape =[number of derivatives wanted/stated, dimensionality]
                porders = [[1,2],[0,1]] ->
                proders[0,0] = order of differentiraion for 1st dimension for the first derivation
                proders[0,1] = order of differentiraion for 2st dimension for the first derivation
                proders[1,0] = order of differentiraion for 1st dimension for the second derivation
                proders[1,1] = order of differentiraion for 2st dimension for the second derivation

    Outputs
    -------
    bmat: 2d array of floats. Contains the RHS vector/matrix. bmat.shape = [number of TSE terms,number of derrivatives to compute]

    """
    TSElen = pows.shape[0]
    dims = pows.shape[1]
    numdfs = porders.shape[0] # number of derivative combinations to compute
    # Compute the factorial terms for each TSE term
    bfacts = axis1fact(pows)
    # Find which combination of powers in "pows" matches the powers given in "porders"
    # Need to check each derivation in porders
    bmat = np.zeros((TSElen,numdfs),dtype=np.float64)
    for i in range(numdfs):
        for j in range(TSElen):
            match = 1
            for m in range(dims):
                if porders[i,m] != pows[j,m]:
                    match = 0
            if match:
                bmat[j,i] = bfacts[j]
    return bmat



def gen_gComponents(coords,porders,N,delta,vols=1):
    """
    Generate the "a" coeffs, powers(xsi_i) of xsi components and weights(xsi) of g(xsi) functions, for a relative function.
    If vols = 1, then the volume of each point is computed assuming we have a regular M dimensional grid
    Inputs
    -------
    coords  : 2d array of floats. Coordinates of points -->coords.shape = [number of points, dimensionality of points].
    porders: 2d array of floats. Contains the orders of differentiation for each dimension.
                porders.shape =[number of derivatives wanted/stated, dimensionality]
                porders = [[1,2],[0,1]] ->
                proders[0,0] = order of differentiraion for 1st dimension for the first derivation
                proders[0,1] = order of differentiraion for 2st dimension for the first derivation
                proders[1,0] = order of differentiraion for 1st dimension for the second derivation
                proders[1,1] = order of differentiraion for 2st dimension for the second derivation

    N : int. Maximum order of TSE terms
    delta : float. Radius of the horizon of point families
    vols :  array of floats. Contains the volume of each point. vols.shape = [number of points].
            If "vols" is not provided in inputs, it is assumed that each points has the same volume and that the domain is regularly discretized.
            The "vols" is thus filled with a single number computed as the product of the discretization step sizes for each dimension

    Outputs
    -------
    ptavecs : 3d array of floats. Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                from the bonds in the family, each point can have different "a" coefficients in its g functions.
                ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms]

    pvec : 2d array of floats. Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
            pvec.shape = [numbers of xsi's/bonds,number of TSE terms]

    weight : 1d array of floats. Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]

    """
    M=coords.shape[1]
    n_points = coords.shape[0]
    D = tse_terms(M,N)-1 #compute the number of tse terms
    pows = tse_pows(N,M,D+1)[1:,:]  # compute the powers of x,y,z,...n, for each term
    fams,start_idx,end_idx,_ = find_neighbors(coords,1.01*delta) # 1) compute the 1D vector of families, 2) the indexes at which each family starts in coords, 3)the indexes at which each family ends in coords, 4)and the numnber of neighbors

    # Amat = np.zeros((D,D),dtype=np.float64)
    # Ainv = np.zeros((D,D),dtype=np.float64)

    ## If the volume of each point is not specified in the inputs, the volume is calculated based on the domain discretization.
    ## It is the product of the disretisation steps in each dimension -> each points has the same volume as if the domain is unifrm
    if not(isinstance(vols,np.ndarray)):
        vols = np.ones(n_points,dtype=np.float64)
        vol=np.float64(1)
        # for m in range(M):
        #     vol =vol * (coords[1,0]-coords[0,0])
        vols[:] = vol

    bmat = gen_bmat(porders,pows)
    ptavecs = np.empty((n_points,porders.shape[0],D),dtype=np.float64)
    pvec = np.zeros((end_idx[-1],D),dtype=np.float64) #end_idx[-1] is the number of xsi there are in the whole domain. fams.shape[0] is larger since we had to assume the maximum number of members for each point(not all families are full!)
    weight = np.zeros(end_idx[-1],dtype=np.float64)

    @njit
    def avecsloop(coords,fams,start_idx,end_idx,delta,vols,bmat,ptavecs,pvec,weight):
        """
        Computes the things that are stated in the outer function. This function just goes through the actual computation, while the outer function
        prepares all the necessary data needed for the coputation. This function should never be called dirrectly
        Inputs:
        --------
        coords      : array of array of floats (coordinates of points=. shape = [number of points, dimensionality of points]. Coordinates should be floats.
        fams        : 1D array of floats (ID numbers of points). Contains the ID's of the neighbors of all points starting with the ID's of neighbors of point 0.
        start_idx   : 1D array of ints. Component "i" of the start_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" start.
        end_idx     : 1D array of ints. Component "i" of the end_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" end.
        delta   : float. Radius of the horizon of point families.
        vols    : array of floats. Contains the volume of each point. vols.shape = [number of points].
                    If "vols" is not provided in inputs, it is assumed that each points has the same volume and that the domain is regularly discretized.
                    The "vols" is thus filled with a single number computed as the product of the discretization step sizes for each dimension
        bmat    : 2d array of floats. Contains the RHS vector/matrix. bmat.shape = [number of TSE terms,number of derrivatives to compute]
        ptavecs : 3d array of floats. Is empty at input. Used as a container to write in
                    Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                    Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                    from the bonds in the family, each point can have different "a" coefficients in its g functions.
                    ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms].

        pvec    : 2d array of floats. Is empty at input. Used as a container to write in
                    Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
                    pvec.shape = [numbers of xsi's/bonds,number of TSE terms]. Is empty at input. USedf as a container to write in

        weight : 1d array of floats. Is empty at input. Used as a container to write in
                 Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]
                 Is empty at input. USedf as a container to write in
        Outputs:
        --------
        ptavecs : 3d array of floats. Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                    Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                    from the bonds in the family, each point can have different "a" coefficients in its g functions.
                    ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms]

        pvec    : 2d array of floats. Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
                    pvec.shape = [numbers of xsi's/bonds,number of TSE terms].

        weight : 1d array of floats. Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]
        """
        for pt in range(n_points): #range(n_points)
            Amat = np.zeros((D,D))
            for ptmemloc in range(start_idx[pt],end_idx[pt]):#:range(start_idx[pt],end_idx[pt])
                xsi = coords[fams[ptmemloc]]-coords[pt]
                ximag = np.sum(xsi*xsi)**0.5 #based on a few custom loops i made, this line is around half the computational cost
                #Calculate the powers of components of xsi based on the TSE term powers
                curpvec = axis1prod(np.power(xsi,pows)) # used my own function because np.prod() is now support by numba if i want to use the axis=1 optional argument. My function is even faster!
                #print(xsi,curpvec)
                #Calculate the weight function for each xsi -> w(xsi)
                curweight = np.exp(-4*(ximag/delta)**2)
                Amat = Amat + np.outer(curpvec,curpvec)*curweight*vols[fams[ptmemloc]]
                pvec[ptmemloc] = curpvec
                weight[ptmemloc] = curweight
                #print(curweight)
            #End of A matrix computation
            #Find the inverse of the shape matrix for point "pt"
            Ainv = matinv(Amat)
            #Write the g function parameters into ptgvecs array
            #print(np.linalg.cond(Amat))
            ptavecs[pt]=np.dot(Ainv,bmat).T
        return ptavecs, pvec, weight
    return avecsloop(coords,fams,start_idx,end_idx,delta,vols,bmat,ptavecs,pvec,weight)

def gen_gComponents2(coords,porders,N,delta,vols=1):
    """
    Generate the "a" coeffs, powers(xsi_i) of xsi components and weights(xsi) of g(xsi) functions, for a relative function.
    IT is different from the first "gen_gComponents" function only in the weight function. The wight here is w_n = (delta/|xsi|)n+1
    If vols = 1, then the volume of each point is computed assuming we have a regular M dimensional grid
    Inputs
    -------
    coords  : 2d array of floats. Coordinates of points -->coords.shape = [number of points, dimensionality of points].
    porders: 2d array of floats. Contains the orders of differentiation for each dimension.
                porders.shape =[number of derivatives wanted/stated, dimensionality]
                porders = [[1,2],[0,1]] ->
                proders[0,0] = order of differentiraion for 1st dimension for the first derivation
                proders[0,1] = order of differentiraion for 2st dimension for the first derivation
                proders[1,0] = order of differentiraion for 1st dimension for the second derivation
                proders[1,1] = order of differentiraion for 2st dimension for the second derivation

    N : int. Maximum order of TSE terms
    delta : float. Radius of the horizon of point families
    vols :  array of floats. Contains the volume of each point. vols.shape = [number of points].
            If "vols" is not provided in inputs, it is assumed that each points has the same volume and that the domain is regularly discretized.
            The "vols" is thus filled with a single number computed as the product of the discretization step sizes for each dimension

    Outputs
    -------
    ptavecs : 3d array of floats. Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                from the bonds in the family, each point can have different "a" coefficients in its g functions.
                ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms]

    pvec : 2d array of floats. Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
            pvec.shape = [numbers of xsi's/bonds,number of TSE terms]

    weight : 1d array of floats. Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]

    """
    M=coords.shape[1]
    n_points = coords.shape[0]
    D = tse_terms(M,N)-1 #compute the number of tse terms
    pows = tse_pows(N,M,D+1)[1:,:]  # compute the powers of x,y,z,...n, for each term
    fams,start_idx,end_idx,_ = find_neighbors(coords,1.01*delta) # 1) compute the 1D vector of families, 2) the indexes at which each family starts in coords, 3)the indexes at which each family ends in coords, 4)and the numnber of neighbors

    # Amat = np.zeros((D,D),dtype=np.float64)
    # Ainv = np.zeros((D,D),dtype=np.float64)

    ## If the volume of each point is not specified in the inputs, the volume is calculated based on the domain discretization.
    ## It is the product of the disretisation steps in each dimension -> each points has the same volume as if the domain is unifrm
    if not(isinstance(vols,np.ndarray)):
        vols = np.ones(n_points,dtype=np.float64)
        vol=np.float64(1)
        # for m in range(M):
        #     vol =vol * (coords[1,0]-coords[0,0])
        vols[:] = vol

    bmat = gen_bmat(porders,pows)
    ptavecs = np.empty((n_points,porders.shape[0],D),dtype=np.float64)
    pvec = np.zeros((end_idx[-1],D),dtype=np.float64) #end_idx[-1] is the number of xsi there are in the whole domain. fams.shape[0] is larger since we had to assume the maximum number of members for each point(not all families are full!)
    weightvecs = np.zeros((end_idx[-1],D),dtype=np.float64)

    powssum = np.sum(pows,axis=1)

    powMat = np.full((D,D),N)
    for i in range(N-1,0,-1):
        for j in np.where(powssum==i)[0]:
            powMat[j,:] = powssum
            powMat[:,j] = powssum

    @njit
    def avecsloop2(coords,fams,start_idx,end_idx,delta,vols,bmat,ptavecs,pvec,powMat,weightvecs):
        """
        Computes the things that are stated in the outer function. This function just goes through the actual computation, while the outer function
        prepares all the necessary data needed for the coputation. This function should never be called dirrectly
        Inputs:
        --------
        coords      : array of array of floats (coordinates of points=. shape = [number of points, dimensionality of points]. Coordinates should be floats.
        fams        : 1D array of floats (ID numbers of points). Contains the ID's of the neighbors of all points starting with the ID's of neighbors of point 0.
        start_idx   : 1D array of ints. Component "i" of the start_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" start.
        end_idx     : 1D array of ints. Component "i" of the end_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" end.
        delta   : float. Radius of the horizon of point families.
        vols    : array of floats. Contains the volume of each point. vols.shape = [number of points].
                    If "vols" is not provided in inputs, it is assumed that each points has the same volume and that the domain is regularly discretized.
                    The "vols" is thus filled with a single number computed as the product of the discretization step sizes for each dimension
        bmat    : 2d array of floats. Contains the RHS vector/matrix. bmat.shape = [number of TSE terms,number of derrivatives to compute]
        ptavecs : 3d array of floats. Is empty at input. Used as a container to write in
                    Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                    Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                    from the bonds in the family, each point can have different "a" coefficients in its g functions.
                    ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms].

        pvec    : 2d array of floats. Is empty at input. Used as a container to write in
                    Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
                    pvec.shape = [numbers of xsi's/bonds,number of TSE terms]. Is empty at input. USedf as a container to write in

        weight : 1d array of floats. Is empty at input. Used as a container to write in
                 Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]
                 Is empty at input. USedf as a container to write in
        Outputs:
        --------
        ptavecs : 3d array of floats. Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                    Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                    from the bonds in the family, each point can have different "a" coefficients in its g functions.
                    ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms]

        pvecs    : 2d array of floats. Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
                    pvec.shape = [numbers of xsi's/bonds,number of TSE terms].

        weight : 1d array of floats. Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]
        """
        for pt in range(n_points): #range(n_points)
            Amat = np.zeros((D,D))
            for ptmemloc in range(start_idx[pt],end_idx[pt]):#:range(start_idx[pt],end_idx[pt])
                xsi = coords[fams[ptmemloc]]-coords[pt]
                ximag = np.sum(xsi*xsi)**0.5 #based on a few custom loops i made, this line is around half the computational cost
                #Calculate the powers of components of xsi based on the TSE term powers
                curpvec = axis1prod(np.power(xsi,pows)) # used my own function because np.prod() is now support by numba if i want to use the axis=1 optional argument. My function is even faster!
                #print(xsi,curpvec)
                #Calculate the weight function for each xsi -> w(xsi)
                curweightMat = np.power((delta/ximag),powMat)
                Amat = Amat + np.outer(curpvec,curpvec)*curweightMat*vols[fams[ptmemloc]]
                pvec[ptmemloc] = curpvec
                weightvecs[ptmemloc] = curweightMat[0,:]
                #print(curweight)
            #End of A matrix computation
            #Find the inverse of the shape matrix for point "pt"
            Ainv = matinv(Amat)
            #Write the g function parameters into ptgvecs array
            #print(np.linalg.cond(Amat))
            ptavecs[pt]=np.dot(Ainv,bmat).T
        return ptavecs, pvec, weightvecs
    return avecsloop2(coords,fams,start_idx,end_idx,delta,vols,bmat,ptavecs,pvec,powMat,weightvecs)

@njit(parallel = True)
def gen_xsigval(ptavecs,pvec,weight,start_index,end_index):
    """
    Calculates the value of g functions for each xsi/bond and stores it in an array so that they dont need to be computed again for the same domain.
    The outputs of "gen_gComponents()" are meant to be input into this function -> ptavecs,pvec,weight = gen_gComponents()
    Inputs
    -------
    ptavecs : 3d array of floats. Contains the values of "a" coefficients in g functions corresponding with the wanted/stated derivatives.
                    Since each point can have a different horizon shape and each point computes its Amat (which together with bmat defines the "a" coefficients),
                    from the bonds in the family, each point can have different "a" coefficients in its g functions.
                    ptavecs.shape = [number points,number of derivatives wanted/stated,number of TSE terms]

    pvec    : 2d array of floats. Contains the values of the product of xsi component powers for each TSE term for each xsi/bond.
                    pvec.shape = [numbers of xsi's/bonds,number of TSE terms]

    weight : 1d array of floats. Contains the values of the weight function for every xsi/bond. weight.shape = [number of xsi's/bonds]

    start_index   : 1D array of ints. Component "i" of the start_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" start.
    end_index     : 1D array of ints. Component "i" of the end_index array shows at which index, in the neighbors array, the neighbors of point with ID = "i" end.

    Outputs:
    --------

    """
    g = np.zeros((end_index[-1],ptavecs.shape[1]),dtype=np.float64)
    for pt in prange(start_index.shape[0]):
        for ptmemloc in range(start_index[pt],end_index[pt]):#range(start_index[pt],end_index[pt]):
            #Calculate the value of g(xsi) for every xsi in domain
            g[ptmemloc] = np.sum(ptavecs[pt] * pvec[ptmemloc] * weight[ptmemloc],axis=1)
    return g

@njit(parallel =True)
def PDDOderive(funvals,fams,g,vols,start_index,end_index):
    """
    """
    res = np.zeros((funvals.shape[0],g.shape[1]),dtype=np.float64)
    for pt in prange(funvals.shape[0]):
        ptres =np.zeros((1,g.shape[1]))
        for ptmemloc in range(start_index[pt],end_index[pt]):
            #Calculate the value of g(xsi) for every xsi in domain
            ptres = ptres+ (funvals[fams[ptmemloc]]-funvals[pt])*g[ptmemloc]*vols[ptmemloc]
            #print(ptres,ptmemloc)

        res[pt] = ptres
    return res

@njit
def calc_bondLenghts(coordVec,neighbors,start_idx,end_idx):
    """
    Function to calculate the length of each bond based od points in coordVec.
    """
    bondLens = np.empty_like(neighbors,dtype=np.float64)
    for pt in range(coordVec.shape[0]):
        for famid in range(start_idx[pt],end_idx[pt]):
            sqsum=0
            for m in range(coordVec.shape[1]):
                sqsum += (coordVec[neighbors[famid],m] - coordVec[pt,m])**2
            curdist = sqsum**0.5
            bondLens[famid] = curdist
    return bondLens

@njit
def find_valueID(arr,val):
    for i in range(arr.shape[0]):
        if arr[i] ==val:
            return i

@njit
def gen_Gmat2D(coordVec,neighbors,start_idx,end_idx,delta):
    """This function is flawed as of the 28/02/2024 github state. The inputs in the Amat are not
    multiplied by the area of ceah PD material point. Therefore the resulting calculation do not give a force density vector."""
    #Calculate b matrix
    bmat = np.zeros((3,3))
    ##g20
    bmat[0,0] = 2
    ##g02
    bmat[1,1] = 2
    ##g11
    bmat[2,2] = 1

    #Calculate the a parameters for g functions of G matrix
    ptavecs = np.zeros((coordVec.shape[0],3,3),dtype = float)
    Amat = np.zeros((3,3),dtype=float)
    xsiweight = float(0)
    pvec = np.zeros((neighbors.shape[0],3))
    weight = np.zeros(neighbors.shape[0])
    Qvec = np.zeros(3,dtype=float)
    for pt in range(coordVec.shape[0]):
        Amat[:,:]=0
        for j in range(start_idx[pt], end_idx[pt]):
            xsi = coordVec[neighbors[j]] - coordVec[pt]
            xsiX = xsi[0]
            xsiY = xsi[1]
            xsimag = (xsiX**2 + xsiY**2)**0.5
            xsiweight = (delta/xsimag)**3
            Amat[0,0] += xsiX**4 * xsiweight
            Amat[0,1] += xsiX**2 * xsiY**2 * xsiweight
            Amat[1,1] += xsiY**4 * xsiweight
            Amat[2,2] += xsiX**2 * xsiY**2 * xsiweight
            pvec[j,0] = xsiX**2
            pvec[j,1] = xsiY**2
            pvec[j,2] = xsiX * xsiY
            weight[j] = xsiweight
        Amat[1,0] = Amat[0,1]
        #Matrix conditioning Q@A@Q @ Qinv@a = Q@b
        Qvec[0] = 1/Amat[0,0]
        Qvec[1] = 1/Amat[1,1]
        Qvec[2] = 1/Amat[2,2]
        Q =np.eye(3)*np.sqrt(Qvec)
        QAmatQ =  Q@Amat@Q
        Qbmat = Q@bmat

        QAmatQinv = np.linalg.inv(QAmatQ)
        ptavecs[pt]=(Q@np.dot(QAmatQinv,Qbmat)).T

    #Create the values of g for each xsi
    G_xsigvals = np.zeros((neighbors.shape[0],ptavecs.shape[1]),dtype=float)
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt], end_idx[pt]):
            G_xsigvals[j] = np.sum(ptavecs[pt] * pvec[j] * weight[j],axis=1)

    return G_xsigvals

@njit
def gen_Gmat2D_fixed(coordVec,neighbors,start_idx,end_idx,delta,area):
    """Non:_fixed function :This function is flawed as of the 28/02/2024 github state. The inputs in the Amat are not
    multiplied by the area of each PD material point. Therefore the resulting calculation do not give a force density vector.
    --------
    This _fixed function has the area part in the calculation. Need to cehck if it is correct.
    """
    #Calculate b matrix
    bmat = np.zeros((3,3))
    ##g20
    bmat[0,0] = 2
    ##g02
    bmat[1,1] = 2
    ##g11
    bmat[2,2] = 1

    #Calculate the a parameters for g functions of G matrix
    ptavecs = np.zeros((coordVec.shape[0],3,3),dtype = float)
    Amat = np.zeros((3,3),dtype=float)
    xsiweight = float(0)
    pvec = np.zeros((neighbors.shape[0],3))
    weight = np.zeros(neighbors.shape[0])
    Qvec = np.zeros(3,dtype=float)
    for pt in range(coordVec.shape[0]):
        Amat[:,:]=0
        for j in range(start_idx[pt], end_idx[pt]):
            xsi = coordVec[neighbors[j]] - coordVec[pt]
            xsiX = xsi[0]
            xsiY = xsi[1]
            xsimag = (xsiX**2 + xsiY**2)**0.5
            xsiweight = (delta/xsimag)**3
            Amat[0,0] += xsiX**4 * xsiweight * area
            Amat[0,1] += xsiX**2 * xsiY**2 * xsiweight * area
            Amat[1,1] += xsiY**4 * xsiweight * area
            Amat[2,2] += xsiX**2 * xsiY**2 * xsiweight * area
            pvec[j,0] = xsiX**2
            pvec[j,1] = xsiY**2
            pvec[j,2] = xsiX * xsiY
            weight[j] = xsiweight
        Amat[1,0] = Amat[0,1]
        #Matrix conditioning Q@A@Q @ Qinv@a = Q@b
        Qvec[0] = 1/Amat[0,0]
        Qvec[1] = 1/Amat[1,1]
        Qvec[2] = 1/Amat[2,2]
        Q =np.eye(3)*np.sqrt(Qvec)
        QAmatQ =  Q@Amat@Q
        Qbmat = Q@bmat

        QAmatQinv = np.linalg.inv(QAmatQ)
        ptavecs[pt]=(Q@np.dot(QAmatQinv,Qbmat)).T

    #Create the values of g for each xsi
    G_xsigvals = np.zeros((neighbors.shape[0],ptavecs.shape[1]),dtype=float)
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt], end_idx[pt]):
            G_xsigvals[j] = np.sum(ptavecs[pt] * pvec[j] * weight[j],axis=1)

    return G_xsigvals

@njit
def gen_Gmat2D_fixed2(coordVec: np.ndarray[float,2], neighbors:np.ndarray[float,1], start_idx:np.ndarray[float,1], end_idx:np.ndarray[float,1], delta:float, area: np.ndarray[float,1]) -> np.ndarray[float,1]:
    """This function is does the same as gen_Gmat2D except for the fact that it takes the volume into account when creating A matrices.
        Also in this implementation of this function, the area of EACH point is used and not a homogenous area across all points.
    --------
    This _fixed function has the area part in the calculation. Need to check if it is correct.
    """
    #Calculate b matrix
    bmat = np.zeros((3,3))
    ##g20
    bmat[0,0] = 2
    ##g02
    bmat[1,1] = 2
    ##g11
    bmat[2,2] = 1

    #Calculate the a parameters for g functions of G matrix
    ptavecs = np.zeros((coordVec.shape[0],3,3),dtype = float)
    Amat = np.zeros((3,3),dtype=float)
    xsiweight = float(0)
    pvec = np.zeros((neighbors.shape[0],3))
    weight = np.zeros(neighbors.shape[0])
    Qvec = np.zeros(3,dtype=float)
    for pt in range(coordVec.shape[0]):
        Amat[:,:]=0
        for j in range(start_idx[pt], end_idx[pt]):
            xsi = coordVec[neighbors[j]] - coordVec[pt]
            xsiX = xsi[0]
            xsiY = xsi[1]
            xsimag = (xsiX**2 + xsiY**2)**0.5
            xsiweight = (delta/xsimag)**3 #* area[neighbors[j]] # I can just multiply with the area here once instead of 4 times below!
            Amat[0,0] += xsiX**4 * xsiweight * area[neighbors[j]]
            Amat[0,1] += xsiX**2 * xsiY**2 * xsiweight * area[neighbors[j]]
            Amat[1,1] += xsiY**4 * xsiweight * area[neighbors[j]]
            Amat[2,2] += xsiX**2 * xsiY**2 * xsiweight * area[neighbors[j]]
            pvec[j,0] = xsiX**2
            pvec[j,1] = xsiY**2
            pvec[j,2] = xsiX * xsiY
            weight[j] = xsiweight
        Amat[1,0] = Amat[0,1]
        #Matrix conditioning Q@A@Q @ Qinv@a = Q@b
        Qvec[0] = 1/Amat[0,0]
        Qvec[1] = 1/Amat[1,1]
        Qvec[2] = 1/Amat[2,2]
        Q =np.eye(3)*np.sqrt(Qvec)
        QAmatQ =  Q@Amat@Q
        Qbmat = Q@bmat

        QAmatQinv = np.linalg.inv(QAmatQ)
        ptavecs[pt]=(Q@np.dot(QAmatQinv,Qbmat)).T

    #Create the values of g for each xsi
    G_xsigvals = np.zeros((neighbors.shape[0],ptavecs.shape[1]),dtype=float)
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt], end_idx[pt]):
            G_xsigvals[j] = np.sum(ptavecs[pt] * pvec[j] * weight[j],axis=1)

    return G_xsigvals


@njit
def gen_Gmat3D(coordVec,neighbors,start_idx,end_idx,delta):
    ""
    #Calculate b matrix
    bmat = np.zeros((6,6))
    ##g200
    bmat[0,0] = 2
    ##g020
    bmat[1,1] = 2
    ##g002
    bmat[2,2] = 2
    ##g110
    bmat[3,3] = 1
    ##g101
    bmat[4,4] = 1
    ##g011
    bmat[5,5] = 1


    #Calculate the a parameters for g functions of G matrix
    ptavecs = np.zeros((coordVec.shape[0],6,6),dtype = float)
    Amat = np.zeros((6,6),dtype=float)
    xsiweight = float(0)
    pvec = np.zeros((neighbors.shape[0],6))
    weight = np.zeros(neighbors.shape[0])
    Qvec = np.zeros(6,dtype=float)
    for pt in range(coordVec.shape[0]):
        Amat[:,:]=0
        for j in range(start_idx[pt], end_idx[pt]):
            xsi = coordVec[neighbors[j]] - coordVec[pt]
            xsiX = xsi[0]
            xsiY = xsi[1]
            xsiZ = xsi[2]
            xsimag = (xsiX**2 + xsiY**2 + xsiZ**2)**0.5
            xsiweight = (delta/xsimag)**3

            Amat[0,0] += xsiX**4 * xsiweight #
            Amat[1,1] += xsiY**4 * xsiweight #
            Amat[2,2] += xsiZ**4 * xsiweight #
            Amat[0,1] += (xsiX*xsiY)**2 * xsiweight #
            Amat[0,2] += (xsiX *xsiZ)**2* xsiweight #
            Amat[1,2] += (xsiY *xsiZ)**2* xsiweight #

            Amat[3,3] += (xsiX*xsiY)**2 * xsiweight #
            Amat[4,4] += (xsiX*xsiZ)**2 * xsiweight #
            Amat[5,5] += (xsiY*xsiZ)**2 * xsiweight #


            pvec[j,0] = xsiX**2
            pvec[j,1] = xsiY**2
            pvec[j,2] = xsiZ**2
            pvec[j,3] = xsiX * xsiY
            pvec[j,4] = xsiX * xsiZ
            pvec[j,5] = xsiY * xsiZ
            weight[j] = xsiweight

        Amat[1,0] = Amat[0,1]
        Amat[2,0] = Amat[0,2]
        Amat[2,1] = Amat[1,2]
        #Matrix conditioning Q@A@Q @ Qinv@a = Q@b
        Qvec[0] = 1/Amat[0,0]
        Qvec[1] = 1/Amat[1,1]
        Qvec[2] = 1/Amat[2,2]
        Qvec[3] = 1/Amat[3,3]
        Qvec[4] = 1/Amat[4,4]
        Qvec[5] = 1/Amat[5,5]
        Q =np.eye(6)*np.sqrt(Qvec)
        QAmatQ =  Q@Amat@Q
        Qbmat = Q@bmat

        QAmatQinv = np.linalg.inv(QAmatQ)
        ptavecs[pt]=(Q@np.dot(QAmatQinv,Qbmat)).T

    #Create the values of g for each xsi
    G_xsigvals = np.zeros((neighbors.shape[0],ptavecs.shape[1]),dtype=float)
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt], end_idx[pt]):
            G_xsigvals[j] = np.sum(ptavecs[pt] * pvec[j] * weight[j],axis=1)

    return G_xsigvals


# ----------- Mehcanics part --------------------------------------------------------#
@njit(parallel = True)
def gen_StiffMat(coordVec,delta,Emod):
    mu = Emod/(2*(float(1+0.25)))
    stiffMat = np.zeros((coordVec.shape[0]*2,coordVec.shape[0]*2),dtype=float)
    neighbors,start_idx,end_idx,n_neighbors = find_neighbors(coordVec,1.01*delta)
    Gvec = gen_Gmat2D(coordVec,neighbors,start_idx,end_idx,delta)
    G11vec = Gvec[:,0]
    G12vec = Gvec[:,2]
    G22vec = Gvec[:,1]
    for pt in prange(coordVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
            #Calculate S for the bond between point "j" in family of "pt"
            G11 = G11vec[j]
            G12 = G12vec[j]
            G22 =  G22vec[j]
            G11G22 = G11+G22
            S11 = G11G22 + 2*G11
            S12 = 2*G12
            S22 = G11G22 + 2*G22

            #Calculate S for the bond petween point "pt" in family of "j"
            j_pt = find_valueID(neighbors[start_idx[neighbors[j]]:end_idx[neighbors[j]]],pt) + start_idx[neighbors[j]]
            jG11 = G11vec[j_pt]
            jG12 = G12vec[j_pt]
            jG22 =  G22vec[j_pt]
            jG11G22 = jG11+jG22
            jS11 = jG11G22 + 2*jG11
            jS12 = 2*jG12
            jS22 = jG11G22 + 2*jG22

            #Calculate Sbar for bond pt_j (pt is main point j in fam member)
            Sbar11 = 0.5*(S11+jS11)
            Sbar12 = 0.5*(S12+jS12)
            Sbar22 = 0.5*(S22+jS22)

            #Sum of contibutions of main point in every Xsi
            stiffMat[pt*2,pt*2] = stiffMat[pt*2,pt*2] - Sbar11
            stiffMat[pt*2,pt*2+1] = stiffMat[pt*2,pt*2+1] - Sbar12
            stiffMat[pt*2+1,pt*2] = stiffMat[pt*2+1,pt*2] - Sbar12
            stiffMat[pt*2+1,pt*2+1] = stiffMat[pt*2+1,pt*2+1] - Sbar22

            #Contributions of family members of main point
            stiffMat[pt*2,2*neighbors[j]] = Sbar11
            stiffMat[pt*2,2*neighbors[j]+1] = Sbar12
            stiffMat[pt*2+1,2*neighbors[j]] = Sbar12
            stiffMat[pt*2+1,2*neighbors[j]+1] = Sbar22

    stiffMat = stiffMat*mu
    return stiffMat

@njit
def gen_StiffMat3D(coordVec,delta,Emod):
    mu = Emod/(2*(float(1+0.25)))
    stiffMat = np.zeros((coordVec.shape[0]*3,coordVec.shape[0]*3),dtype=float)
    neighbors,start_idx,end_idx,n_neighbors = find_neighbors(coordVec,1.01*delta)
    Gvec = gen_Gmat3D(coordVec,neighbors,start_idx,end_idx,delta)
    G11vec = Gvec[:,0]
    G22vec = Gvec[:,1]
    G33vec = Gvec[:,2]
    G12vec = Gvec[:,3]
    G13vec = Gvec[:,4]
    G23vec = Gvec[:,5]
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
            #Calculate S for the bond between point "j" in family of "pt"
            G11 = G11vec[j]
            G12 = G12vec[j]
            G13 = G13vec[j]
            G22 = G22vec[j]
            G23 = G23vec[j]
            G33 = G33vec[j]

            G11G22G33 = G11+G22+G33
            S11 = G11G22G33 + 2*G11
            S22 = G11G22G33 + 2*G22
            S33 = G11G22G33 + 2*G33
            S12 = 2*G12
            S13 = 2*G13
            S23 = 2*G23

            #Calculate S for the bond petween point "pt" in family of "j"
            j_pt = find_valueID(neighbors[start_idx[neighbors[j]]:end_idx[neighbors[j]]],pt) + start_idx[neighbors[j]]
            jG11 = G11vec[j_pt]
            jG12 = G12vec[j_pt]
            jG13 = G13vec[j_pt]
            jG22 =  G22vec[j_pt]
            jG23 =  G23vec[j_pt]
            jG33 =  G33vec[j_pt]

            jG11G22G33 = jG11+jG22+jG33
            jS11 = jG11G22G33 + 2*jG11
            jS22 = jG11G22G33 + 2*jG22
            jS33 = jG11G22G33 + 2*jG33
            jS12 = 2*jG12
            jS13 = 2*jG13
            jS23 = 2*jG23


            #Calculate Sbar for bond pt_j (pt is main point j in fam member)
            Sbar11 = 0.5*(S11+jS11)
            Sbar12 = 0.5*(S12+jS12)
            Sbar13 = 0.5*(S13+jS13)
            Sbar22 = 0.5*(S22+jS22)
            Sbar23 = 0.5*(S23+jS23)
            Sbar33 = 0.5*(S33+jS33)

            #Sum of contibutions of main point in every Xsi
            stiffMat[pt*3,pt*3] = stiffMat[pt*3,pt*3] - Sbar11
            stiffMat[pt*3,pt*3+1] = stiffMat[pt*3,pt*3+1] - Sbar12
            stiffMat[pt*3,pt*3+2] = stiffMat[pt*3,pt*3+2] - Sbar13

            stiffMat[pt*3+1,pt*3] = stiffMat[pt*3+1,pt*3] - Sbar12
            stiffMat[pt*3+1,pt*3+1] = stiffMat[pt*3+1,pt*3+1] - Sbar22
            stiffMat[pt*3+1,pt*3+2] = stiffMat[pt*3+1,pt*3+2] - Sbar23

            stiffMat[pt*3+2,pt*3] = stiffMat[pt*3+2,pt*3] - Sbar13
            stiffMat[pt*3+2,pt*3+1] = stiffMat[pt*3+2,pt*3+1] - Sbar23
            stiffMat[pt*3+2,pt*3+2] = stiffMat[pt*3+2,pt*3+2] - Sbar33

            #Contributions of family members of main point
            stiffMat[pt*3,3*neighbors[j]] = Sbar11
            stiffMat[pt*3,3*neighbors[j]+1] = Sbar12
            stiffMat[pt*3,3*neighbors[j]+2] = Sbar13

            stiffMat[pt*3+1,3*neighbors[j]] = Sbar12
            stiffMat[pt*3+1,3*neighbors[j]+1] = Sbar22
            stiffMat[pt*3+1,3*neighbors[j]+2] = Sbar23

            stiffMat[pt*3+2,3*neighbors[j]] = Sbar13
            stiffMat[pt*3+2,3*neighbors[j]+1] = Sbar23
            stiffMat[pt*3+2,3*neighbors[j]+2] = Sbar33

    stiffMat = stiffMat*mu
    return stiffMat

@njit
def applyDispBC(BCvec,stiffnessMat,RHSvec):
    """Apply displacement boundary conditions to the global stiffnes matrix and RHS vector.
    Inputs:
    -------
    BCvec : 2d array of floats. Contains the points for which displacement BC will be applied to and the boundary conditions that will be applied to them. BCvec.shape = [number of constrained points, number of constrained DOF]
                BCvec.shape[0] -> contains point ID's from the discretization. POints with the ID writen here get constrained
                BCvec.shape[1] -> for 2d, this dimension is of length 2 or 3. The first input in the ID's, the second is the X direction constraints and third the Y direction constraints.

    stiffnessMat : 2d array of floats. The stiffness matrix created using the "pddo.gen_StiffMat()" function. stiffnessMat.shape = [number of points in discretization * 2, number of points in discretization * 2] = [number of DOF, number of DOF] -> for 2D
                    This stiffness matrix does not have any BC applied to it!

    RHSvec : 1d array of floats. Array of zeros to be used to construct the RHS vector with applied BC's. RHSvec.shape = [number of DOF]

    Outputs:
    --------
    BCstiffnessMat : 2d array of floats. The stiffness matrix from the inputs but with BC's applied! BCstiffnessMat.shape = [number of DOF,number of DOF]

    RHSvec : 1d array of floats. The RHS vector from the inputs but with applied BC's. RHSvec.shape = [number of DOF]

    """
    dim = 0
    if BCvec.shape[1] == 3: # 3 because the first (zeroth in python) column are ID's
        dim = 2
    elif BCvec.shape[1] == 4: # 4 because the first (zeroth in python) column are ID's
        dim = 3
    else: print("BCvec is of wrong dimensions.")

    if (stiffnessMat.shape[1] == RHSvec.shape[0]):
        BCpts  = BCvec[:,0]
        BCdofs = BCvec.shape[1]-1 #This is the number of DOF which are constrained in the BCs
        BCstiffnessMat = stiffnessMat
        RHSvec2 = np.zeros_like(RHSvec)
        for i in range(BCpts.shape[0]):
            pt = np.int64(BCpts[i])
            for locdof in range(BCdofs):
                globdof = np.int64(pt*dim + locdof)
                RHSvec2cur = BCstiffnessMat[:,globdof] * BCvec[i,locdof+1]

                #Modification of the RHS Vector to be added to the RHS vector outside of the outer loop
                RHSvec2 = RHSvec2 + RHSvec2cur
                RHSvec[globdof] = BCvec[i,locdof+1]

                #Modify the stiffnes matrix
                BCstiffnessMat[:,globdof] = 0
                BCstiffnessMat[globdof,:] = 0
                BCstiffnessMat[globdof,globdof] = np.float64(1)

        RHSvec = RHSvec - RHSvec2

        for i in range(BCpts.shape[0]):
            pt = np.int64(BCpts[i])
            for locdof in range(BCdofs):
                globdof = pt*dim + locdof
                RHSvec[globdof] = BCvec[i,locdof+1]

        return BCstiffnessMat,RHSvec

    else:
        print("Dimensions of stiffness matrix do not match dimensions of the RHS vector!")

def applyDispBC2(BCvec: np.ndarray[float,2], stiffnessMat:  np.ndarray[float,2], RHSvec: np.ndarray[float,1], dim:int):
    """Apply displacement boundary conditions to the global stiffnes matrix and RHS vector.
        Works for many combined BCloads together.
    Inputs:
    -------
    BCvec : 2d array of floats. Contains the points for which displacement BC will be applied to and the boundary conditions that will be applied to them. BCvec.shape = [number of constrained points, number of constrained DOF]
                BCvec[:,0] -> contains point ID's from the discretization. Points with the ID writen here get constrained
                BCvec[:,1:4] -> contains bool's to see if the DOF is contrained. BCvec[:,1] -> DOF 0, BCvec[:,2] -> DOF 1, BCvec[:,2] -> DOF 3.
                BCvec[:,4:7] -> contains the value of the applied displacement for DOF's from BCvec[:,1:4]

    stiffnessMat : 2d array of floats. The stiffness matrix created using the "pddo.gen_StiffMat()" function. stiffnessMat.shape = [number of points in discretization * 2, number of points in discretization * 2] = [number of DOF, number of DOF] -> for 2D
                    This stiffness matrix does not have any BC applied to it!

    RHSvec : 1d array of floats. Array of zeros to be used to construct the RHS vector with applied BC's. RHSvec.shape = [number of DOF]

    Outputs:
    --------
    BCstiffnessMat : 2d array of floats. The stiffness matrix from the inputs but with BC's applied! BCstiffnessMat.shape = [number of DOF,number of DOF]

    RHSvec : 1d array of floats. The RHS vector from the inputs but with applied BC's. RHSvec.shape = [number of DOF]

    """

    if (stiffnessMat.shape[1] == RHSvec.shape[0]):
        BCpts  = BCvec[:,0]
        BCdofs = dim #This is the number of DOF which are constrained in the BCs
        BCstiffnessMat = stiffnessMat
        RHSvec2 = np.zeros_like(RHSvec)
        for i in range(BCpts.shape[0]):
            pt = np.int64(BCpts[i])
            for locdof in range(BCdofs):
                locdof_isFixed = bool(BCvec[i,locdof+1])
                if locdof_isFixed:
                    globdof = np.int64(pt*dim + locdof)
                    RHSvec2cur = BCstiffnessMat[:,globdof] * BCvec[i,locdof+4]

                    #Modification of the RHS Vector to be added to the RHS vector outside of the outer loop
                    RHSvec2 = RHSvec2 + RHSvec2cur
                    RHSvec[globdof] = BCvec[i,locdof+4]

                    #Modify the stiffnes matrix
                    BCstiffnessMat[:,globdof] = 0
                    BCstiffnessMat[globdof,:] = 0
                    BCstiffnessMat[globdof,globdof] = np.float64(1)

        RHSvec = RHSvec - RHSvec2

        for i in range(BCpts.shape[0]):
            pt = np.int64(BCpts[i])
            for locdof in range(BCdofs):
                globdof = pt*dim + locdof
                RHSvec[globdof] = BCvec[i,locdof+4]

        return BCstiffnessMat,RHSvec

    else:
        print("Dimensions of stiffness matrix do not match dimensions of the RHS vector!")

@njit
def calc_bond_normals(pd_point_count, pd_bond_count, coordVec, neighbors, start_idx, end_idx):
        normals = np.empty(shape=(pd_bond_count,2))
        i=0
        for point in range(pd_point_count):
            for bondid in range(start_idx[point],end_idx[point]):
                #Calculate relative vector between teo points
                _rel_vec = (coordVec[point] - coordVec[neighbors[bondid]])
                normal = _rel_vec/(_rel_vec[0]**2 + _rel_vec[1]**2)**0.5
                normals[i] = normal
                i += 1
        return normals

        """Solve the current incremental load for equilibrium using Newton-Rhapson method."""

        if num_max_it < 0 or type(num_max_it) != int:
            print("The maximum number of iterations can not be a negative value and it must be an integer type!")

        if epsilon <= 0:
            print("Epsilon can not be a negative value! Epsilon == 0 is not realistic and must be larger! (0 < epsilon)")

        #This for loop will need to be changed for a while loop eventually!
        _RHSvec = np.zeros(self.geometry.coordVec.shape[0]*2)
        _newCoordVec = self.geometry.coordVec
        for iter in range(num_max_it):# and error > epsilon:
            print(f"Iteration {iter}")
            #Create stiffness matrix and external force vector for current displacements
            _stiffmat = self.gen_stiffness_matrix(self.geometry.curLiveBonds, self.geometry.curBondDamage)
            #Apply boudary conditions
            if iter ==0:
                _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)
            else:
                BCvec[:,1:] = 0
                _BC_stiffmat,_BC_RHSvec = self.apply_displacement_BC(BCvec,_stiffmat,_RHSvec)

            _BC_stiffmatCSR = csr_matrix(_BC_stiffmat)
            #Solve sistem of equation for disaplcements
            _solu = spsolve(_BC_stiffmatCSR,_BC_RHSvec)
            _disps = np.reshape(_solu,(int(_solu.shape[0]/2),2))
            _newCoordVec = _newCoordVec + _disps
            #Calculate bond stretches
            _cur_bond_stretches = self.calc_bond_stretches(_newCoordVec)
            # print(_cur_bond_stretches)
            self.geometry.curBondDamage = self.update_bond_damage(np.abs(_cur_bond_stretches),s1,sc)
            #Calculate the internal forces based on the calculated stretches
            internal_bondforce = np.zeros(shape=2)
            internal_pointforces = np.zeros_like(_BC_RHSvec)
            i=0
            for point in range(self.geometry.pd_point_count):
                #Sum the force contributions of all points int the family based on stretches
                internal_bondforce = internal_bondforce*0
                for bondid in range(self.geometry.start_idx[point],self.geometry.end_idx[point]):
                    internal_bondforce += self.calc_int_bond_force_from_stretch(self.material.damage_model,_cur_bond_stretches[bondid],self.geometry.bond_normals[bondid],self.geometry.curBondDamage[bondid],self.geometry.delta,self.geometry.delta)
                internal_pointforces[i] = internal_bondforce[0]
                internal_pointforces[i+1] = internal_bondforce[1]
                i += 1
            #Calculate the residual of the forces (Residual_forces = F_external - F_internal)
            _residual_forces = (_BC_RHSvec - internal_pointforces)
            # print(_BC_RHSvec)
            # print(internal_pointforces)
            # print(_residual_forces)
            # Calculate the norm of the residual and check if it is smaller than some defined value for epsilon
            # _residual_forces_norm = np.linalg.norm(_residual_forces/internal_pointforces)
            dif_vec = _residual_forces.reshape((np.int(_residual_forces.shape[0]/2),2)) - internal_pointforces.reshape((np.int(internal_pointforces.shape[0]/2),2))
            _residual_forces_norm = np.linalg.norm(np.sum(dif_vec,axis=0)/np.sum(internal_pointforces.reshape((np.int(internal_pointforces.shape[0]/2),2))))
            #If residual is small enough return the displacements
            if _residual_forces_norm <= epsilon:
                print(f"Convergence succesfull in step: {iter}. Maximum bond stretch = {_cur_bond_stretches.max()}")
                return _disps
        print(f"Convergence was not achieved! Residual = {_residual_forces_norm}")
        return _disps
@njit
def _calc_bond_damage(cur_bondStretches:np.ndarray, s1:float, sc:float) -> np.ndarray:
    new_damage = np.zeros(cur_bondStretches.shape[0])
    for bond in range(cur_bondStretches.shape[0]):
        if  s1 <= cur_bondStretches[bond] <= sc:
            new_damage[bond] = sc/(sc-s1)*(1- s1/cur_bondStretches[bond])
        if cur_bondStretches[bond] >= sc:
            new_damage[bond] = 1
    return new_damage


@njit
def _generate_stiffness_matrix(coordVec,neighbors, start_idx, end_idx, G11vec, G12vec,G22vec, mu, LiveBonds, Damage, stiffMat):
    """Function is only meant to be called inside the "gen_stiffness_matrix" method of the PDFatigueModel class!
    """
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
            currentBondStatus = LiveBonds[j]
            alive = True
            if currentBondStatus == alive:
                #Calculate S for the bond between point "j" in family of "pt"
                G11 = G11vec[j]
                G12 = G12vec[j]
                G22 =  G22vec[j]
                G11G22 = G11+G22
                S11 = G11G22 + 2*G11
                S12 = 2*G12
                S22 = G11G22 + 2*G22

                #Calculate S for the bond petween point "pt" in family of "j"
                j_pt = find_valueID(neighbors[start_idx[neighbors[j]]:end_idx[neighbors[j]]],pt) + start_idx[neighbors[j]]
                jG11 = G11vec[j_pt]
                jG12 = G12vec[j_pt]
                jG22 =  G22vec[j_pt]
                jG11G22 = jG11+jG22
                jS11 = jG11G22 + 2*jG11
                jS12 = 2*jG12
                jS22 = jG11G22 + 2*jG22

                #Calculate Sbar for bond pt_j (pt is main point j in fam member)
                Sbar11 = (1-Damage[j])*0.5*(S11+jS11)
                Sbar12 = (1-Damage[j])*0.5*(S12+jS12)
                Sbar22 = (1-Damage[j])*0.5*(S22+jS22)

                #Sum of contibutions of main point in every Xsi
                stiffMat[pt*2,pt*2] = stiffMat[pt*2,pt*2] - Sbar11
                stiffMat[pt*2,pt*2+1] = stiffMat[pt*2,pt*2+1] - Sbar12
                stiffMat[pt*2+1,pt*2] = stiffMat[pt*2+1,pt*2] - Sbar12
                stiffMat[pt*2+1,pt*2+1] = stiffMat[pt*2+1,pt*2+1] - Sbar22

                #Contributions of family members of main point
                stiffMat[pt*2,2*neighbors[j]] = Sbar11
                stiffMat[pt*2,2*neighbors[j]+1] = Sbar12
                stiffMat[pt*2+1,2*neighbors[j]] = Sbar12
                stiffMat[pt*2+1,2*neighbors[j]+1] = Sbar22

    stiffMat = stiffMat * mu
    return stiffMat

@njit
def _generate_stiffness_matrix2(coordVec,neighbors, start_idx, end_idx, G11vec, G12vec,G22vec, mu, LiveBonds, Damage, stiffMat):
    """Function is only meant to be called inside the "gen_stiffness_matrix" method of the PDFatigueModel class!
    """
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
            currentBondStatus = LiveBonds[j]
            alive = True
            if currentBondStatus == alive:
                #Calculate S for the bond between point "j" in family of "pt"
                G11 = G11vec[j]
                G12 = G12vec[j]
                G22 =  G22vec[j]
                G11G22 = G11+G22
                S11 = G11G22 + 2*G11
                S12 = 2*G12
                S22 = G11G22 + 2*G22

                #Calculate S for the bond petween point "pt" in family of "j"
                j_pt = find_valueID(neighbors[start_idx[neighbors[j]]:end_idx[neighbors[j]]],pt) + start_idx[neighbors[j]]
                jG11 = G11vec[j_pt]
                jG12 = G12vec[j_pt]
                jG22 =  G22vec[j_pt]
                jG11G22 = jG11+jG22
                jS11 = jG11G22 + 2*jG11
                jS12 = 2*jG12
                jS22 = jG11G22 + 2*jG22

                #Calculate Sbar for bond pt_j (pt is main point j in fam member)
                Sbar11 = (1-Damage[j])*0.5*(S11+jS11) * mu[j]
                Sbar12 = (1-Damage[j])*0.5*(S12+jS12) * mu[j]
                Sbar22 = (1-Damage[j])*0.5*(S22+jS22) * mu[j]

                #Sum of contibutions of main point in every Xsi
                stiffMat[pt*2,pt*2] = stiffMat[pt*2,pt*2] - Sbar11
                stiffMat[pt*2,pt*2+1] = stiffMat[pt*2,pt*2+1] - Sbar12
                stiffMat[pt*2+1,pt*2] = stiffMat[pt*2+1,pt*2] - Sbar12
                stiffMat[pt*2+1,pt*2+1] = stiffMat[pt*2+1,pt*2+1] - Sbar22

                #Contributions of family members of main point
                stiffMat[pt*2,2*neighbors[j]] = Sbar11
                stiffMat[pt*2,2*neighbors[j]+1] = Sbar12
                stiffMat[pt*2+1,2*neighbors[j]] = Sbar12
                stiffMat[pt*2+1,2*neighbors[j]+1] = Sbar22

    stiffMat = stiffMat
    return stiffMat

@njit
def _generate_bond_stiffnesses(coordVec,neighbors, start_idx, end_idx, G11vec, G12vec,G22vec, mu) -> np.ndarray[float,2]:
    """Function is only meant to be called inside the "gen_stiffness_matrix" method of the PDFatigueModel class!
    """
    bond_stiff_matrix_array = np.zeros(shape=(neighbors.shape[0],2,2))
    for pt in range(coordVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
                #Calculate S for the bond between point "j" in family of "pt"
                G11 = G11vec[j]
                G12 = G12vec[j]
                G22 =  G22vec[j]
                G11G22 = G11+G22
                S11 = G11G22 + 2*G11
                S12 = 2*G12
                S22 = G11G22 + 2*G22

                #Calculate S for the bond petween point "pt" in family of "j"
                j_pt = find_valueID(neighbors[start_idx[neighbors[j]]:end_idx[neighbors[j]]],pt) + start_idx[neighbors[j]]
                jG11 = G11vec[j_pt]
                jG12 = G12vec[j_pt]
                jG22 =  G22vec[j_pt]
                jG11G22 = jG11+jG22
                jS11 = jG11G22 + 2*jG11
                jS12 = 2*jG12
                jS22 = jG11G22 + 2*jG22

                #Calculate Sbar for bond pt_j (pt is main point j in fam member)
                Sbar11 = 0.5*(S11+jS11) * mu[j]
                Sbar12 = 0.5*(S12+jS12) * mu[j]
                Sbar22 = 0.5*(S22+jS22) * mu[j]


                bond_stiff_matrix_array[j,0,0] = Sbar11
                bond_stiff_matrix_array[j,0,1] = Sbar12
                bond_stiff_matrix_array[j,1,0] = Sbar12
                bond_stiff_matrix_array[j,1,1] = Sbar22

    return bond_stiff_matrix_array

@njit
def _generate_bond_displacement_vecs(dispVec, neighbors, start_idx, end_idx):

    del_disps = np.zeros((neighbors.shape[0],2))
    for pt in range(dispVec.shape[0]):
        for j in range(start_idx[pt],end_idx[pt]):
            del_disps[j] = dispVec[neighbors[j]] - dispVec[pt]

    return del_disps

@njit
def generate_force_dens_vecs(bond_stiffness_mat:np.ndarray[float,3], bond_displacement_vecs:np.ndarray[float,2]) -> np.ndarray:
    force_dens_vecs = np.zeros(shape = bond_displacement_vecs.shape)
    for bond in range (bond_displacement_vecs.shape[0]):

        force_dens_vecs[bond,0] = bond_stiffness_mat[bond,0,0] * bond_displacement_vecs[bond,0] + bond_stiffness_mat[bond,0,1] * bond_displacement_vecs[bond,1]
        force_dens_vecs[bond,1] = bond_stiffness_mat[bond,1,0] * bond_displacement_vecs[bond,0] + bond_stiffness_mat[bond,1,1] * bond_displacement_vecs[bond,1]

    return force_dens_vecs

def family_integration(neighbors: np.ndarray[float,1], start_idx:np.ndarray[int,1], end_idx:np.ndarray[int,1], bond_values:np.ndarray[float,1],point_volumes: np.ndarray[float,1]) -> np.ndarray[float,1]:
    pt_sum = np.empty_like(point_volumes)
    for pt in range(point_volumes.shape[0]):
        cur_sum = 0
        for j in range(start_idx[pt],end_idx[pt]):
            cur_sum = cur_sum + bond_values[j] * point_volumes[neighbors[j]]
        pt_sum[pt] = cur_sum
    return pt_sum