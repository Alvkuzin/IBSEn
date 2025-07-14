import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import interp1d
from ibsen.utils import loggrid, interplg
from math import ceil

def solveTransport(a1, a2, Q, x_grid, y_grid,
                     xcond = 0, ycond = 0, parall = False,
                     a1_args = (), a2_args = (), Q_args = (),
                    t_max=10.0, max_step=0.01,
                    tol = 1e-8):
    """
    Solves steady-state transport equation:
      a1(x) dF/dx + a2(y) dF/dy = Q(x,y)
    using the trace of characteristics method.

    Parameters
    ----------
    a1 : callable
        Function a1(x, *a1_args) returning scalar or array of shape (len(x),).
    a2 : callable
        Function a2(y, *a2_args) returning scalar or array of shape (len(y),).
    Q : callable
        Function Q(x,y, *Q_args) returning source term on the grid.
    x_cond : float
        Where to pose a zero boundary condition on x-axis
    y_cond : float
        Where to pose a zero boundary condition on y-axis
    x_grid : 1D array
        Uniform grid in x-direction.
    y_grid : 1D array
        Grid in y-direction (can be non-uniform).
    a1_args: tuple, optional
        Extra arguments to pass to a1
    a2_args: tuple, optional
        Extra arguments to pass to a20.01
    Q_args: tuple, optional
        Extra arguments to pass to Q
    t_max: float, optional
        The max value of a coordinate along the characteristics to which it
        would be traced (t_span = [0, t_max] is passed to solve_ivp).
        Default is 10
    max_step: float, optional
        Passed to solve_ivp as max_step. Default is 0.01.
    tol: float, optioal
        Passed to solve_ivp as rtol. Default is 1e-8.

    Returns
    -------
    F : 2D array
        Numeric solution F(x_i, y_j) of shape (len(x_grid), len(y_grid)).
    """
    # --------------------------------------------------
    # 1) Characteristic integrator for a single point (x0, y0)
    # --------------------------------------------------
    def compute_F_point(x0, y0):
        """
        Compute F(x0, y0) by tracing backward characteristic:
        dX/dt = -a1(X),  dY/dt = -a2(Y),  dI/dt = Q(X, Y)
        until X= xcond or Y = ycond, with I(0)=0.
        Returns I(t_end) = F(x0, y0).x0, y0, t_max=10.0, max_step=0.01
        """
        def odes(t, vars):
            x, y, I = vars
            return [-a1(x, *a1_args), -a2(y, *a2_args), Q(x, y, *Q_args)]

        # Event when characteristic hits x=0
        def hit_x0(t, vars):
            return vars[0] - xcond
        hit_x0.terminal = True
        hit_x0.direction = -1

        # Event when characteristic hits y=0
        def hit_y0(t, vars):
            return vars[1] - ycond
        hit_y0.terminal = True
        hit_y0.direction = -1

        sol = solve_ivp(
            fun=odes,
            t_span=(0, t_max),
            y0=[x0, y0, 0.0],
            events=[hit_x0, hit_y0],
            max_step=max_step,
            rtol = tol,
            # atol = tol
        )

        return sol.y[2, -1]

    # --------------------------------------------------
    # 2) Solve on a grid: method of characteristics
    # --------------------------------------------------
    F_num = np.zeros((len(x_grid), len(y_grid)))
    for i, x in enumerate(x_grid):
        if parall:
            def func_parall(j):
                y = y_grid[j]
                if x == xcond or y == ycond:
                    return  0.0
                else:
                    return compute_F_point(x, y)
            res = Parallel(n_jobs = 10)(delayed(func_parall)(j_y) for j_y in range(y_grid.size))
            F_num[i, :] = np.array(res)
        if not parall:
            for j, y in enumerate(y_grid):
                if x == xcond or y == ycond:
                    F_num[i, j] = 0.0
                else:
                    F_num[i, j] = compute_F_point(x, y)
    return F_num


def solveTranspFDM(a1, a2, Q, x_grid, y_grid,
                     a1_args = (), a2_args = (), Q_args = (),
                    conserv = False, bound = 'dir'):
    """
    Solves steady-state transport equation:
      a1(x,y) dF/dx + a2(x,y) dF/dy = Q(x,y) or
      a1(x,y) dF/dx + d(F * a2(x,y))/dy = Q(x,y)
    using upwind finite differences on grid (x_grid, y_grid).

    Parameters
    ----------
    a1 : callable
        Function a1(x,y, *a1_args) returning scalar or array of shape (len(x),len(y)).
    a2 : callable
        Function a2(x,y, *a2_args) returning scalar or array of shape (len(x),len(y)).
    Q : callable
        Function Q(x,y, *Q_args) returning source term on the grid.
    x_grid : 1D array
        Uniform grid in x-direction.
    y_grid : 1D array
        Grid in y-direction (can be non-uniform).
    a1_args: tuple, optional
        Extra arguments to pass to a1
    a2_args: tuple, optional
        Extra arguments to pass to a2
    Q_args: tuple, optional
        Extra arguments to pass to Q
    conserv: bool, optional
        What equation to solve. If conserv = False: default, the equation
        a1(x,y) dF/dx + a2(x,y) dF/dy = Q(x,y) is solved. If conserv = True,
        the equation a1(x,y) dF/dx + d(F * a2(x,y))/dy = Q(x,y) is solved.
    bound: str, optional
        What boudary conditions to apply. If bound = 'dir': default, then
        Dirichlet conditions are imposed at min(x_grid), max(x_grid), 
        min(y_grid), max(y_grid). If bound = 'neun', then Neumann-type 
        conditions are used at min(x_grid) and min(y_grid), all other 
        conditions are Dirichlet-type. All boundary conditions are zero.

    Returns
    -------
    F : 2D array
        Numeric solution F(x_i, y_j) of shape (len(x_grid), len(y_grid)).
    """
    Nx = len(x_grid)
    Ny = len(y_grid)
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    dx = x_grid[1] - x_grid[0]
    # Precompute non-uniform dy for upwinding
    dy_backward = np.empty(Ny)
    dy_forward = np.empty(Ny)
    dy_backward[0] = np.nan
    for j in range(1, Ny):
        dy_backward[j] = y_grid[j] - y_grid[j-1]
    for j in range(Ny-1):
        dy_forward[j] = y_grid[j+1] - y_grid[j]
    dy_forward[-1] = np.nan

    # Evaluate velocities and source on grid
    A1 = a1(X, Y, *a1_args)
    A2 = a2(X, Y, *a2_args)
    Q = Q(X, Y, *Q_args)

    # Assemble sparse system A F = b
    N = Nx * Ny
    def idx(i, j):
        return i * Ny + j

    rows, cols, data, b = [], [], [], []
    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)
            # Boundary: Dirichlet condition from Q_func or user
            if i == 0 or j == Ny-1 or j == 0:
                rows.append(k); cols.append(k); data.append(1.0)
                # b.append(Q(x_grid[i], y_grid[j], bc=True))
                b.append(0)
                continue
            if bound == 'dir':
                if i == Nx-1:
                    rows.append(k); cols.append(k); data.append(1.0)
                    # b.append(Q(x_grid[i], y_grid[j], bc=True))
                    b.append(0)
                    continue
                
            if bound == 'neun':
                if i == Nx-1:
                    # ∂u/∂x = 0 => (u[1,j] - u[0,j])=0
                    rows += [k, k]
                    cols += [idx(Nx-2,j), k]
                    data += [1.0, -1.0]
                    b.append(0.0)
                    continue
                # if j == 0:
                #     # ∂u/∂y = 0 => (u[i,1] - u[i,0])=0
                #     rows += [k, k]
                #     cols += [idx(i,1), k]
                #     data += [1.0, -1.0]
                #     b.append(0.0)
                #     continue
 
                
            a1_ij = A1[i, j]
            a2_ij = A2[i, j]
            diag = 0.0
            rhs = Q[i, j]

            # x upwind
            if a1_ij >= 0:
                coef = a1_ij / dx
                diag += coef
                rows.append(k); cols.append(idx(i-1, j)); data.append(-coef)
            else:
                coef = -a1_ij / dx
                diag += coef
                rows.append(k); cols.append(idx(i+1, j)); data.append(-coef)
            if not conserv:
                # Advective term in y: a2 ∂u/∂y
                if a2_ij >= 0:
                    dyb = dy_backward[j]
                    coef = a2_ij / dyb
                    diag += coef
                    rows.append(k); cols.append(idx(i, j-1)); data.append(-coef)
                else:
                    dyf = dy_forward[j]
                    coef = -a2_ij / dyf
                    diag += coef
                    rows.append(k); cols.append(idx(i, j+1)); data.append(-coef)
            if conserv:
                # Conservative term in y: ∂(a2 u)/∂y
                a2_ij = A2[i,j]
                # downward flux interface at j-1/2
                if a2_ij >= 0:
                    dy = dy_backward[j]
                    coef_in = A2[i,j-1] / dy
                    rows.append(k); cols.append(idx(i,j-1)); data.append(-coef_in)
                    coef_out = a2_ij / dy
                    diag += coef_out
                else:
                    dy = dy_forward[j]
                    coef_in = -A2[i,j+1] / dy
                    rows.append(k); cols.append(idx(i,j+1)); data.append(-coef_in)
                    coef_out = -a2_ij / dy
                    diag += coef_out

            # diagonal and RHS
            rows.append(k); cols.append(k); data.append(diag)
            b.append(rhs)

    A_mat = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    b_vec = np.array(b)
    F_vec = spla.spsolve(A_mat, b_vec)

    return F_vec.reshape((Nx, Ny))



def solve_for_n(v_func, edot_func, f_func, v_args, edot_args, f_args,
                 s_grid, e_grid, e_cond = 0, parall = False, tol = 1e-8, 
                 method = 'FDM_cons', bound = 'dir'):
    """
    Solves steady-state transport equation:
      v_func(s) dn/ds d(edot_func(e) * n)/de = f_func(s,e) or
      v_func(s) dn/ds d(edot_func(s, e) * n)/de = f_func(s,e)
    on grid (s_grid, e_grid).

    Parameters
    ----------
    v_func : callable
        Function v_func(s, *a1_args) returning scalar or array of shape (len(s),).
    a2 : callable
        If method = '' Function a2(x,y, *a2_args) returning scalar or array of shape (len(x),len(y)).
    Q : callable
        Function Q(x,y, *Q_args) returning source term on the grid.
    x_grid : 1D array
        Uniform grid in x-direction.
    y_grid : 1D array
        Grid in y-direction (can be non-uniform).
    a1_args: tuple, optional
        Extra arguments to pass to a1
    a2_args: tuple, optional
        Extra arguments to pass to a2
    Q_args: tuple, optional
        Extra arguments to pass to Q

    bound: str, optional
        What boudary conditions to apply. If bound = 'dir': default, then
        Dirichlet conditions are imposed at min(x_grid), max(x_grid), 
        min(y_grid), max(y_grid). If bound = 'neun', then Neumann-type 
        conditions are used at min(x_grid) and min(y_grid), all other 
        conditions are Dirichlet-type. All boundary conditions are zero.

    Returns
    -------
    F : 2D array
        Numeric solution F(x_i, y_j) of shape (len(x_grid), len(y_grid)).
    """    
    # --------------------------------------------------
    # 1) for N = n * edot, solve the usual equation 
    # v_func dN/ds + edot_func dN/de = f_func * edot_func
    # --------------------------------------------------
    def Q_rhs(x, y):
        return f_func(x, y, *f_args) * edot_func(y, *edot_args)
    def a1_(x, y):
        return v_func(x, *v_args)
    def a2_(x, y):
        return edot_func(x, y, *edot_args)
    if method == 'char':
        Nbig = solveTransport(a1 = v_func, a2 = edot_func, Q = Q_rhs, 
            x_grid = s_grid, y_grid = e_grid, parall = parall,
            a1_args = v_args, a2_args = edot_args, Q_args = (), ycond = e_cond,
            tol = tol)
        ss, ee = np.meshgrid(s_grid, e_grid, indexing = 'ij')
        return Nbig / edot_func(ee, *edot_args)
    if method == 'FDM':

        Nbig = solveTranspFDM(a1 = a1_, a2 = a2_, Q = Q_rhs, 
            x_grid = s_grid, y_grid = e_grid, bound = bound)
        ss, ee = np.meshgrid(s_grid, e_grid, indexing = 'ij')
        return Nbig / edot_func(ee, *edot_args)
    if method == 'FDM_cons':
        nsmall = solveTranspFDM(a1 = a1_, a2 = a2_, Q = f_func,
        a1_args = (), a2_args = (), Q_args = f_args,    
            x_grid = s_grid, y_grid = e_grid, conserv = True, bound = bound)
        return nsmall
    
    
    

def Denys_solver(t_evol, edot_func, Q_func, emin = 6e8, emax=5e14,
                 overshoot_time = 1e5, step_shortest_cool_time = 1e-1,
                 edot_args = (), Q_args = (),
                 injection_rate = 3e32, test_energies = None, parall = False):
    """
    A code for solving the time-dependent energy transfer equation
    dn/dt + d(Edot n)/dt = Q.
    The idea is the following. 
    
    Stage 1: for ONE electron with maximum energy,
    solve the equation dE/dt = Edot and find the solution E(t). This solution
    can be inverted to yield a function t_evol(E).
    
    Stage 2: to find how ONE energy bin of energy E0 with N0 particles evolve,
    we first evolve all N0 particles from max energy to E0 for a time 
    t_evol(E0). Then we evolve it for aonther real evolution time t_evol, while
    at the same time injecting particles with a rate Q, constant for t_evol
    (so that the number of injected particles is simply growing linearly with
     time from 0 to t_evol). Thus, for one energy bin, we obtain a histogram 
    of (N0 + Q(E0) * t_evol) particles distributed along some energies.
    
    Stage 3: We perform Stage 2 for all energies. 
    
    For now, we start with zero initial conditions: n(t=0, E) = 0.
    
    Functions edot and Q are ONLY e-dependent! That is, edot=edot(e, *edot_args),
    Q = Q(e, q_args).
    

    Parameters
    ----------
    t_evol : float
        For what time to evolve.
    edot_func : callable
        An energy loss [eV/s] function edot_func(1d-arr, *edot_args) -> 1d-arr.
    Q_func : TYPE
        An injection [1/s] function Q_func(1d-arr, *Q_args) -> 1d-arr.
    emin : float, optional
        Min energy [eV] for creating test_energeis is it's not provided.
        The default is 6e8.
    emax : float, optional
        Max energy [eV] for creating test_energeis is it's not provided.
        The default is 5e14.
    overshoot_time : float, optional
        The time for which the one-electron dependence E(t) should be 
        tabulated. The default is 1e5.
    step_shortest_cool_time : float, optional
        The multiplicator for the shortest cooling time. The default is 1e-1.
    edot_args : tuple, optional
        Optional arguments for Edot-function. The default is ().
    Q_args : tuple, optional
        Optional arguments for Q-function. The default is ().
    injection_rate : float, optional
        Multiplicator for Q-function, should you need it. The default is 3e32.
    test_energies : 1d-array, optional
        DESCRIPTION. The default is None.
    parall : Bool, optional
        Whether to parallel calculations. The default is False.

    Returns
    -------
    TYPE
        ee [eV], dN/de [1/eV]: energies of bin centres and spectrum.

    """
    # edot = lambda e_: edot_func(e_, *edot_args)
    # Q = lambda e_: Q_func(e_, *Q_args)
    edot = edot_func
    Q = Q_func
    if overshoot_time < 1.05 * t_evol:
        overshoot_time = 1.05 * t_evol
    # -------------------------------------------------------------------------
    #                               Stage 1
    # -------------------------------------------------------------------------
    
    if test_energies is None:
        test_energies = loggrid(x1 = emin, x2 = emax, n_dec = 61)
    
    # injection_rate : electron @emin: electrons/s/eV
    ee = np.max(test_energies)
    tcool_beginning = - np.min( test_energies / edot(test_energies) )
    #first step
    tt = 0
    dt = min( [step_shortest_cool_time * tcool_beginning,
               0.1*overshoot_time ])
    # ninj = injection_rate * dt
    
    #tt += dt
    all_energies = [ ee ]
    all_times = [ 0 ]
    
    #all time steps
    mindt = dt #minimal evolution timescale
    t_since_emin = 0
    # start_time = time.time()
    while( t_since_emin <= overshoot_time ):
        tcool = -ee/edot(ee)
        dt = min( [step_shortest_cool_time * tcool, 0.1*overshoot_time ])
        dE = dt * edot(ee)
        ee += dE
        all_energies.append( ee )
        if( ee<=emin ): t_since_emin += dt
        if( dt < mindt): mindt = dt
        tt += dt
        all_times.append (tt)
        if( ee<emin/6): break # we dont care for low energies
    all_times = np.array(all_times)
    all_energies = np.array( all_energies )
    
    # logspl_t_e = interp1d(np.log10(all_times), np.log10(all_energies))
    # logspl_back_e_t = interp1d(np.log10(all_energies[::-1]), np.log10(all_times[::-1]))
    
    

    # -------------------------------------------------------------------------
    #                           Stage 2: 
    # -------------------------------------------------------------------------
    

    log_all_times = np.log(all_times)
    log_all_energies = np.log(all_energies) #all interpolation in log-space to keep precision

    func_t_e = lambda t_: np.exp( np.interp( np.log(t_), log_all_times, log_all_energies) )
    func_e_t = lambda e_:  np.exp( np.interp( 
                                  np.log(e_),
                                  log_all_energies[::-1], log_all_times[::-1] 
                                  ) 
                       )

    #evolve whole spectrum
    spec_energies = test_energies[np.logical_and(test_energies> emin, test_energies<emax) ] #energies at which electrons spectrum is defined

    injection_energies = ( spec_energies[1:] * spec_energies[:-1] )**0.5 #actual energies at which electrons are injected
    des = spec_energies[1:] - spec_energies[:-1]
    
    #rates = ElectronInjectionSpectrum(injection_energies) * des # number of injected electrons per second at injection_energies
    #multiplication of spectrum by dE is not accurate. Replaced with integral over dE
    # LESHA::::::: it's we just calculating the flux (for e!) in each band???...
    # LESHA::::::: basically, to know the histogram of how much particles are in a band
    rates = np.zeros(des.shape)
    # erates = np.zeros(des.shape)
    for ee in range(len(spec_energies)-1):
        e1 = spec_energies[ee] 
        e2 = spec_energies[ee+1]
        rates[ee] = injection_rate*quad(Q, e1, e2, limit=10000)[0]# electrons/s ;1e12 since injection_rate is /eV and quad is TeV
        # erates[ee] = injection_rate*quad(Q, e1, e2, limit=10000)[0]

    
    def Evolve1Energy(eidx, show_time):
        """ 
        Evolves a spec for a time show_time. 
        Currently, the initial electron spec (functional form, rates, and grid foe E)
        is defined outside of this function.
        # All electrons injected at emax, but some of them evolving longer.
        # Thus, each electron first evolves for t_offsets[eidx], which 
        # effectively brings an electron to the energy inection_energries[eidx],
        # and then all electrons additionally evolve for a time of evolution.
        # This time of evolution is mimicked by linspace(0, show_time), since
        # we want electrons to evolve for ALL times between 0 and show_time
        """
        # mindt_for_this_e = np.exp( np.interp(np.log(injection_energies[eidx]),
                                      # log_all_energies[::-1], np.log(mindt)[::-1] ) )
        mindt_for_this_e = mindt        
        ninjections = ceil( show_time / mindt_for_this_e) 
        # t_offsets = np.exp( np.interp( 
        #                               np.log(injection_energies),
        #                               log_all_energies[::-1], log_all_times[::-1] 
        #                               ) 
        #                    )
        t_offsets = func_e_t(injection_energies)
        # t_offsets = interplg(x = injection_energies, xdata = all_energies[::-1],
                             # ydata = all_times[::-1])
        # t_offsets = 10**logspl_back_e_t(np.log10(injection_energies))
        norm = mindt_for_this_e*rates[eidx]
        # print(ninjections)
        evolve_for = t_offsets[eidx] + np.linspace(0, show_time, ninjections )  
        # final_energies = np.exp( np.interp( np.log(evolve_for), log_all_times, log_all_energies) )
        final_energies = func_t_e(evolve_for)
        # final_energies = interplg(x = evolve_for, xdata = all_times, ydata=all_energies)
        # final_energies = 10**logspl_t_e(np.log10(evolve_for))
        vals, edgs = np.histogram(final_energies, bins=test_energies)
        return norm*vals, edgs #evolved spectrum of delta-function continuously injected at spec_energies[eidx]

    # -------------------------------------------------------------------------
    #                          Stage 3 and final
    # -------------------------------------------------------------------------

    """
    just for our conveniance, I leave an option of paralleling the
    calculations for 4 cores
    """
    if parall:
        def Lesha_func(iii):
            vals_here, edgs = Evolve1Energy(iii, t_evol)
            return vals_here
        # n_cores = multiprocessing.cpu_count()
        res= Parallel(n_jobs=10)(delayed(Lesha_func)(iii) for iii in range(0, len(spec_energies)-1 ))
        res=np.array(res)
        vals = np.sum(res, axis=0)
    else:
        first = True
        all_rates = [] #E^2 dN/dE/dt for electrons
        all_eavs = []
        for ee in range( len(spec_energies)-1 ):
            # LESHA: literally dN[in a band] * E(band) **2 / dE[band] ????...
            all_rates.append(rates[ee] * injection_energies[ee]**2 / des[ee]) 
            all_eavs.append(injection_energies[ee])

            if( first ):
                vals, edgs = Evolve1Energy(ee, t_evol)
                first = False
            else:
                vals0, edgs = Evolve1Energy(ee, t_evol)
                vals += vals0 # [vals] -- number of electrons in corrresponding energy bin (defined by edgs)
    # norm = 1
    xx = ( test_energies[1:] * test_energies[:-1])**0.5 #edgs correspond to test _energies anyway
    dxx = ( test_energies[1:] - test_energies[:-1])
    return xx, vals / dxx #E   and   dN/dE
    

# def Denys_solver_continuous()

def nonstat_1zone_solver(
    time_grid,
    e_grid,
    Edot_func,
    T_func,
    Q_func,
    n_e1,
    n_e2,
    n_t0
):
    """
    Solve the non-stationary transport equation:
        dn/dt + d(Edot * n)/de + n/T = Q
    using Crank–Nicolson (2nd-order in time) with an upwind finite-volume
    discretization on a non-uniform energy grid.

    Parameters
    ----------
    time_grid : array_like, shape (Nt,)
        Monotonic time points.
    e_grid : array_like, shape (Ne,)
        Energy grid centers (non-uniform).
    Edot : callable
        Edot(e, t) -> (Ne,) array of energy-loss rates.
    T_func : callable
        T_func(e, t) -> (Ne,) array of decay timescales.
    Q_func : callable
        Q_func(e, t) -> (Ne,) array of source terms.

    Returns
    -------
    n : ndarray, shape (Nt, Ne)
        Particle distribution at all times and energies.
    """
    # Arrays
    t_arr = np.asarray(time_grid)
    e_arr = np.asarray(e_grid)
    Nt, Ne = len(t_arr), len(e_arr)

    # Non-uniform cell widths (finite-volume)
    de_plus = np.empty(Ne)
    de_minus = np.empty(Ne)
    de_plus[:-1] = e_arr[1:] - e_arr[:-1]
    de_plus[-1] = de_plus[-2]
    de_minus[1:] = de_plus[:-1]
    de_minus[0] = de_minus[1]
    # Control volume width
    de_center = 0.5 * (de_plus + de_minus)

    # Initialize solution
    n = np.zeros((Nt, Ne))
    
    # Set initial conditions
    n[0, :] = n_t0(e_arr)
    
    # Identity
    I = sp.eye(Ne, format='csc')

    # Time-stepping
    for j in range(Nt - 1):
        dt = t_arr[j+1] - t_arr[j]
        t0, t1 = t_arr[j], t_arr[j+1]
        tm = 0.5 * (t0 + t1)

        # Evaluate mid-step source
        Qm = Q_func(e_arr, tm)

        def build_A(t):
            # Build tridiagonal A such that A @ n = d(Edot*n)/de + n/T
            main = np.zeros(Ne)
            upper = np.zeros(Ne-1)

            E = Edot_func(e_arr, t)
            Tval = T_func(e_arr, t)

            # Interior cells
            for i in range(1, Ne-1):
                # face-centered Edot
                E_im = 0.5 * (E[i-1] + E[i])
                E_ip = 0.5 * (E[i] + E[i+1])
                # divergence: (E_ip * n_{i+1} - E_im * n_i) / Δe_i
                main[i] = -E_im / de_center[i] + 1.0 / Tval[i]
                upper[i] = E_ip / de_center[i]

            # Boundary rows: enforce n=0
            main[0] = main[-1] = 1.0
            # Assemble sparse
            return sp.diags(
                diagonals=[main, upper, np.zeros(Ne-1)],
                offsets=[0, 1, -1],
                format='csc'
            )

        A0 = build_A(t0)
        A1 = build_A(t1)

        # Crank–Nicolson matrices
        LHS = I + 0.5 * dt * A1
        RHS = I - 0.5 * dt * A0

        # RHS vector
        b = RHS.dot(n[j]) + dt * Qm
        b[0] = n_e1(t1)
        b[-1] = n_e2(t1)

        # Advance
        n[j+1] = spla.spsolve(LHS, b)

    return n
    
# def nonstat_1zone_solver_new(e_grid, time_grid, Q_func, T_func, Edot_func,
#                     n_e1, n_e2, n_t0):
#     """
#     Solve dn/dt + d(Edot*n)/de + n/T = Q using backward-Euler in time (implicit) and
#     second-order central differences on a non-uniform energy grid, leveraging sparse
#     linear solves for efficiency.

#     Parameters
#     ----------
#     e_grid : 1D array of floats, non-uniform energy grid of length Ne
#     t_grid : 1D array of floats, uniform time grid of length Nt
#     Q : function Q(e, t) -> array_like or scalar
#     T : function T(e, t) -> array_like or scalar
#     Edot : function Edot(e, t) -> array_like or scalar
#     n_e1 : function n_e1(t) -> Dirichlet BC at e_grid[0]
#     n_e2 : function n_e2(t) -> Dirichlet BC at e_grid[-1]
#     n_t0 : function n_t0(e) -> initial condition at t_grid[0]

#     Returns
#     -------
#     n : 2D array of shape (Nt, Ne)
#         solution values n[t_index, e_index]
#     """
#     Ne = len(e_grid)
#     Nt = len(time_grid)
#     dt = time_grid[1] - time_grid[0]

#     # Precompute spacings
#     h_minus = np.empty(Ne)
#     h_plus  = np.empty(Ne)
#     for i in range(1, Ne):
#         h_minus[i] = e_grid[i] - e_grid[i-1]
#     for i in range(0, Ne-1):
#         h_plus[i] = e_grid[i+1] - e_grid[i]

#     # Allocate solution array and set IC
#     n = np.zeros((Nt, Ne), dtype=float)
#     n[0, :] = n_t0(e_grid)

#     # Time-stepping
#     for m in range(Nt-1):
#         t_new = time_grid[m+1]
#         n0_new = n_e1(t_new)
#         nN_new = n_e2(t_new)

#         # Nint = Ne - 2
#         data = []
#         rows = []
#         cols = []
#         rhs = np.zeros(Ne)
#         Qi_new_arr = Q_func(e_grid, t_new)
#         Ti_new_arr = T_func(e_grid, t_new)
#         Edoti_new_arr = Edot_func(e_grid, t_new)

#         for idx in range(Ne):
#             if idx == 0:
#                 # Dirichlet BC at e_min
#                 rows.append(0)
#                 cols.append(0)
#                 data.append(1.0)
#                 rhs[0] = n0_new
#             elif idx == Ne - 1:
#                 # Dirichlet BC at e_max
#                 rows.append(Ne - 1)
#                 cols.append(Ne - 1)
#                 data.append(1.0)
#                 rhs[-1] = nN_new
#             else:
#                 hm = h_minus[idx]
#                 hp = h_plus[idx]

#                 Qi = Qi_new_arr[idx]
#                 Ti = Ti_new_arr[idx]
#                 Edoti = Edoti_new_arr[idx]

#                 ai = -hp / (hm * (hm + hp))
#                 bi = (hp - hm) / (hm * hp)
#                 ci = hm / (hp * (hm + hp))

#                 Aii = 1.0 / dt + 1.0 / Ti + bi * Edoti
#                 Aim1 = ai * Edoti_new_arr[idx-1]
#                 Aip1 = ci * Edoti_new_arr[idx+1]

#                 # Fill matrix row for idx
#                 rows += [idx, idx, idx]
#                 cols += [idx - 1, idx, idx + 1]
#                 data += [Aim1, Aii, Aip1]

#                 rhs[idx] = Qi + n[m, idx] / dt


#         # build sparse matrix (size Nint x Nint)
#         A = sp.csr_matrix((data, (rows, cols)), shape=(Ne, Ne))

#         # solve
#         n_new = spla.spsolve(A, rhs)

#         # assign
#         n[m+1, :] = n_new
# 
    # return n


def nonstat_characteristic_solver(t_evol, test_energies,
                          edot_func, Q_func,                          
                          init_cond = 0.):
    """
    Solve dn/dt + d(Edot*n)/dE = Q via method of characteristics,
    for time-independent Edot(E) and Q(E), some given initial cond.

    Same signature as Denys_solver, but uses analytic characteristic integration:


    Returns:
    --------
    xx : mid-energy of bins (len = len(test_energies)-1)
    n     : dN/dE at each bin center, shape (len(xx),)
    """
    # 1. define energy grid

    # bin centers and edges
    edges = test_energies
    centers = np.sqrt(edges[:-1] * edges[1:])

    # 2. build fine E-grid for tau and characteristic
    # use edges as nodes


    # # 3. compute tau(E) = integral from E[0] to E of dE'/speed
    
    E = edges
    speed = np.abs(edot_func(E))
    # reverse the grid so that cumulative_trapezoid integrates from E_max → E_min
    tau_rev =cumulative_trapezoid(1./speed[::-1], E[::-1], initial=0)
    # then flip back:
    tau = tau_rev[::-1]
    tau_max = tau[-1]

    # 4. build inverse mapping E_of_tau

    # 5. compute Q(E) along characteristic
    Q_E = Q_func(E)
    Edot_E = edot_func(E)
    # Q vs tau: Q(tau_i) = Q(E_i)

    # 6. cumulative integral G(tau)= \int_0^tau Q d tau'
    G = np.concatenate(([0.0], cumulative_trapezoid(Q_E * Edot_E, tau))) 
    G_of_tau = interp1d(tau, G, kind='linear',# fill_value=(0.0, G[-1]), 
                        assume_sorted=True)

    # 7. compute n at bin centers: for each center energy E_c
    # tau_c = tau(E_c) via interpolation
    tau_of_E = interp1d(E, tau, kind='linear', 
                        #fill_value='extrapolate',
                        assume_sorted=True)
    E_of_tau = interp1d(tau, E, kind='linear',
                        #fill_value='extrapolate', 
                        assume_sorted=True)
 
        

    tau_c = tau_of_E(centers)
    tau_end = tau_c + t_evol
    # clamp to tau_max
    tau_end = np.minimum(tau_end, tau_max)

    # n(E_c) = injection_rate * [G(tau_end) - G(tau_c)]
    G_inject = (G_of_tau(tau_end) - G_of_tau(tau_c))
    # n_vals = injection_rate * G_
    
    if isinstance(init_cond, float):
        e0_grid = edges
        n0_grid = np.zeros(edges.size) + init_cond
    elif (isinstance(init_cond, tuple) or isinstance(init_cond, list)):
        # print('im here')
        e0_grid, n0_grid = init_cond
    else: 
        raise ValueError('not suitable init cond: should be either float OR (e0_grid, n0_grid)')
    
    n0_of_E   = interp1d(e0_grid, n0_grid,
                     kind='linear', fill_value=0.0,
                     bounds_error=False)

    # 1) ages at current centers
    tau_c = tau_of_E(centers)       
    tau0  = tau_c + t_evol
    
    # 2) only those with tau0 <= tau_max have an IC contribution
    ic_mask = tau0 <= tau_max
    
    # 3) back‐track to initial energy E0
    E0 = np.empty_like(centers)
    E0[ic_mask] = E_of_tau(tau0[ic_mask])
    
    # 4) compute IC term in numerator
    G_init = np.zeros_like(centers)
    G_init[ic_mask] = (edot_func(E0[ic_mask])
                     * n0_of_E(E0[ic_mask]))
    
    # 5) total solution
    n_vals = (G_init + G_inject) / edot_func(centers)

        
    return centers, n_vals 
