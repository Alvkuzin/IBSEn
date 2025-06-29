import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad


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
                 overshoot_time = 1e5, step_shortest_cool_time = 1e-3,
                 edot_args = (), Q_args = (),
                 injection_rate = 3e32, test_energies = False, parall = False):
    """
    Should write the  documentation here
    """
    edot = lambda e_: edot_func(e_, *edot_args)
    Q = lambda e_: Q_func(e_, *Q_args)
    if overshoot_time < 1.05 * t_evol:
        overshoot_time = 1.05 * t_evol
    # -------------------------------------------------------------------------
    #                               Stage 1
    # -------------------------------------------------------------------------
    
    # injection_rate : electron @emin: electrons/s/eV
    ee = emax
    tcool_beginning = - emax / edot(emax) 
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
    # print('Denys method takes ---%s--- sec'%(time.time() - start_time))
    # mindt_realmin = mindt
    # mindt = all_times * step_shortest_cool_time*10 # here mindt is overwritten, now, it's an array
    # mindt = mindt + mindt_realmin
    
    # -------------------------------------------------------------------------
    #                           Stage 2: Two Towers
    # -------------------------------------------------------------------------
    
    # if not test_energies:
        # test_energies = np.logspace(np.log10(emax/10), np.log10(emax), 300)
    log_all_times = np.log(all_times)
    log_all_energies = np.log(all_energies) #all interpolation in log-space to keep precision

    #evolve whole spectrum
    spec_energies = test_energies[np.logical_and(test_energies> emin, test_energies<emax) ] #energies at which electrons spectrum is defined

    injection_energies = ( spec_energies[1:] * spec_energies[:-1] )**0.5 #actual energies at which electrons are injected
    des = spec_energies[1:] - spec_energies[:-1]
    
    #rates = ElectronInjectionSpectrum(injection_energies) * des # number of injected electrons per second at injection_energies
    #multiplication of spectrum by dE is not accurate. Replaced with integral over dE
    # LESHA::::::: it's we just calculating the flux (for e!) in each band???...
    # LESHA::::::: basically, to know the histogram of how much particles are in a band
    rates = np.zeros(des.shape)
    erates = np.zeros(des.shape)
    for ee in range(len(spec_energies)-1):
        e1 = spec_energies[ee] 
        e2 = spec_energies[ee+1]
        rates[ee] = injection_rate*quad(Q, e1, e2, limit=10000)[0]# electrons/s ;1e12 since injection_rate is /eV and quad is TeV
        erates[ee] = injection_rate*quad(Q, e1, e2, limit=10000)[0]
        
    E_Q = lambda e_ev: Q(e_ev) * e_ev
    int_spec = quad(E_Q, spec_energies[0], 
                    spec_energies[-1], limit=10000, epsabs=1e-10, 
                    epsrel=1e-10) # TeV^2
    # print int_spec
    int_spec = int_spec[0] * injection_rate*1.6e12 # erg/s
    # int_spec1 = np.ma.sum(erates)
    
    # sed_init_e = rates*injection_energies**2/des
    # N_init_e = trapezoid(sed_init_e / injection_energies**2, injection_energies) #integral of dN/dE
    
    # -------------------------------------------------------------------------
    #                          Stage 3 and final
    # -------------------------------------------------------------------------
    
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
        ninjections = int( show_time / mindt_for_this_e) 
        t_offsets = np.exp( np.interp( np.log(injection_energies),
                                      log_all_energies[::-1], log_all_times[::-1] ) )
        norm = mindt_for_this_e*rates[eidx]
        print(ninjections)
        evolve_for = t_offsets[eidx] + np.linspace(0, show_time, ninjections )  
        final_energies = np.exp( np.interp( np.log(evolve_for), log_all_times, log_all_energies) )
        vals, edgs = np.histogram(final_energies, bins=test_energies)
        return norm*vals, edgs #evolved spectrum of delta-function continuously injected at spec_energies[eidx]

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
    if not parall:
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
    
    
    