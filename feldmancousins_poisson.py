import numpy as np
from scipy.stats import poisson, norm
from scipy.optimize import bisect
import warnings

__all__ = ['mu_acc', 'FC_ints_raw', 'FC_ints', 'FC_ints_search']

def _mu_acc_sca(mu_range, n, b, alpha=0.1):
    print("entering mu_acc_sca")
    b_out = np.zeros_like(mu_range, dtype=bool) # "b_out" = "binary array output"
    n0_min = poisson.ppf(alpha/10,mu_range.min()+b).astype(int) - 2
    n0_min = max(n0_min, 0)
    n0_min = min(n0_min, n)
    n0_max = poisson.ppf(1-alpha/10,mu_range.max()+b).astype(int)
    n0_max = max(n0_max, n)
    n0 = np.r_[n0_min:(n0_max+1)]
    m_inds = np.tile(np.r_[:len(mu_range)].astype(np.int64),(len(n0),1)).T
    n0m = np.tile(n0,(len(mu_range),1))
    mum = np.tile(mu_range,(len(n0),1)).T
    Tmu = np.empty((len(mu_range),len(n0)), dtype=np.float64)
    cut_ngeqb = n0m >= b
    Tmu[cut_ngeqb] = \
        2*(n0m[cut_ngeqb]*np.log(n0m[cut_ngeqb]/(mum[cut_ngeqb]+b))-n0m[cut_ngeqb]+mum[cut_ngeqb]+b)
    Tmu[~cut_ngeqb] = 2*(n0m[~cut_ngeqb]*np.log(b/(mum[~cut_ngeqb]+b))+mum[~cut_ngeqb])
    inds_sort = Tmu.argsort(axis=1)
    Pn0 = poisson.pmf(n0m, mum+b)
    Pn0_s = Pn0[m_inds, inds_sort]
    n0m_s = n0m[m_inds, inds_sort]
    Pn0_s_cum = Pn0_s.cumsum(axis=1)
    Pn0_geq90_cum = (Pn0_s_cum >= (1-alpha)).cumsum(axis=1)
    bool_n_acc = Pn0_geq90_cum < 2
    n0m_s_min = n0m_s.copy()
    n0m_s_min[~bool_n_acc] = int(1e9)
    n_min_acc = n0m_s_min.min(axis=1)
    n0m_s_max = n0m_s.copy()
    n0m_s_max[~bool_n_acc] = -1
    n_max_acc = n0m_s_max.max(axis=1)
    b_out = (n >= n_min_acc) & (n <= n_max_acc)
    return b_out

def _mu_acc_bvec(mu_range, n, b, alpha=0.1):
    print("entering mu_acc_bvec")
    # assume here that the b's will be closely spaced and can use the same n0 range
    b_out = np.zeros_like(mu_range, dtype=bool) # "b_out" = "binary array output"
    n0_min = poisson.ppf(alpha/10,mu_range.min()+b.min()).astype(int) - 2
    n0_min = max(n0_min, 0)
    n0_min = min(n0_min, n)
    n0_max = poisson.ppf(1-alpha/10,mu_range.max()+b.max()).astype(int)
    n0_max = max(n0_max, n)
    n0 = np.r_[n0_min:(n0_max+1)]
    m_dims = (len(b), len(mu_range), len(n0))
    m_inds_1d = np.r_[:len(mu_range)].astype(np.int64)
    m_inds = np.broadcast_to(m_inds_1d[np.newaxis,:,np.newaxis],m_dims)
    b_inds_1d = np.r_[:len(b)].astype(np.int64)
    b_inds = np.broadcast_to(b_inds_1d[:,np.newaxis,np.newaxis],m_dims)
    n0m = np.broadcast_to(n0[np.newaxis,np.newaxis,:],m_dims)
    mum = np.broadcast_to(mu_range[np.newaxis,:,np.newaxis],m_dims)
    Tmu = np.empty(m_dims, dtype=np.float64)
    b_m = np.broadcast_to(b[:,np.newaxis,np.newaxis],m_dims)
    cut_ngeqb = n0m >= b_m
    Tmu[cut_ngeqb] = \
        2*(n0m[cut_ngeqb]*np.log(n0m[cut_ngeqb]/(mum[cut_ngeqb]+b_m[cut_ngeqb])) - \
        n0m[cut_ngeqb]+mum[cut_ngeqb]+b_m[cut_ngeqb])
    Tmu[~cut_ngeqb] = 2*(n0m[~cut_ngeqb]*np.log(b_m[~cut_ngeqb]/(mum[~cut_ngeqb]+b_m[~cut_ngeqb]))+mum[~cut_ngeqb])
    inds_sort = Tmu.argsort(axis=-1)
    Pn0 = poisson.pmf(n0m, mum+b_m)
    Pn0_s = Pn0[b_inds,m_inds, inds_sort]
    n0m_s = n0m[b_inds, m_inds, inds_sort]
    Pn0_s_cum = Pn0_s.cumsum(axis=-1)
    Pn0_geq90_cum = (Pn0_s_cum >= (1-alpha)).cumsum(axis=-1)
    bool_n_acc = Pn0_geq90_cum < 2
    n0m_s_min = n0m_s.copy()
    n0m_s_min[~bool_n_acc] = int(1e9)
    n_min_acc = n0m_s_min.min(axis=-1)
    n0m_s_max = n0m_s.copy()
    n0m_s_max[~bool_n_acc] = -1
    n_max_acc = n0m_s_max.max(axis=-1)
    b_out = (n >= n_min_acc) & (n <= n_max_acc)
    return b_out

def mu_acc(mu_range, n, b, alpha=0.1):
    """
    Given a single observation (n) and a background expectation (b), determine
    which values of mu in mu_range are accepted and which are rejected.
    
    The definition of alpha follows the more traditional one in the literature,
    though the literature is not always consistent on this.  Namely:
    90% confidence intervals correspond to alpha = 0.10
    95% confidence intervals correspond to alpha = 0.05
    etc.
    Inputs:
        mu_range: 1d numpy array of floats.
               n: A single integer
               b: A float, either scalar or 1d array of values.
    Outputs:
           b_out: A boolean numpy array the same size as mu_range.  An element with
                  True means the corresponding element in mu_range is accepted at
                  the given alpha, while False means it is rejected.
                  If input 'b' was a scalar, then b_out is a 1d array the same size
                  as mu_range.  If input 'b' was a vector, then b_out is a 2d array
                  of shape (p, q), with p=len(b) and q=len(mu_range).
    """
    if not isinstance(mu_range, np.ndarray) and (mu_range.ndim != 1):
        raise TypeError("Input 'mu_range' must be a 1d numpy array")
    if not (mu_range.dtype == np.dtype('float64')):
        raise TypeError("mu_range must have dtype = np.float64")
    if not isinstance(n, int):
        raise TypeError("Input 'n' must be a single integer")
    if not isinstance(b, (float, np.ndarray)):
        raise TypeError("Input 'b' must be either a float or an numpy array")
    if isinstance(b, np.ndarray) and (b.ndim != 1):
        raise TypeError("If 'b' is an array, it must be 1d")
    
    b_is_array = isinstance(b, np.ndarray)
    
    if b_is_array:
        outArr = np.empty((len(b), len(mu_range)))
        for k, bb in enumerate(b):
            outArr[k, :] = _mu_acc_sca(mu_range, n, bb, alpha=alpha)
        #return _mu_acc_bvec(mu_range, n, b, alpha=alpha)
        return outArr
    else:
        return _mu_acc_sca(mu_range, n, b, alpha=alpha)
    return -1

def FC_ints_raw(n, b, alpha=0.1, verbosity=0):
    """
    n is the observed number of events
    b is the background expectation.
    Returns a tuple: min_mu and max_mu
    """
    assert isinstance(n, int) or isinstance(n,np.int64), "Input 'n' must be an integer"
    assert isinstance(b, float) or isinstance(b,np.float64), "Input 'b' must be a float"
    m = norm.ppf(1-alpha/2)
    b_eff = max(b, 1e-10) # the method barfs if b is actually zero
    if verbosity >= 1:
        print(f'{b = }', flush=True)
    muR_min = .25*((m-np.sqrt((m**2)+4*n))**2)-b_eff-2
    muR_min = max(0., muR_min)
    muR_max = .25*((m+np.sqrt((m**2)+4*n))**2)-b_eff+10.
    muR_max = max(10., muR_max)
    if verbosity >= 1:
        print(f'mu_range = ({muR_min}, {muR_max})')
    if (muR_max - muR_min)/0.005 < 8000.:
        mu_range = np.arange(muR_min, muR_max, 0.005)
    else:
        mu_range = np.linspace(muR_min, muR_max, 8000)
    #print(f"mu_range: {len(mu_range)} elements")
    mu_pass = mu_range[mu_acc(mu_range, n, b_eff, alpha=alpha)]
    mu_min, mu_max = mu_pass.min(), mu_pass.max()
    if verbosity >= 1:
        print(f'LL, UL: ({mu_min}, {mu_max})')
    if ((muR_min-mu_min)**2 <= (0.02**2)) and (muR_min != 0.):
        wm = f"min of mu pinned at min of mu search range. {muR_min = }|||| {mu_min = }\n" + \
            f"n, b = {n}, {b:0.2f}"
        #warnings.warn(f"min mu pinned at min of mu search range: n, b = {n}, {b:0.2f}", 
        warnings.warn(wm, category=RuntimeWarning)
    if (muR_max-mu_max)**2 <= (0.02**2):
        wm = f"max mu pinned at max of mu search range. {muR_max = }|||| {mu_max = }\n" + \
            f"n, b = {n}, {b:0.2f}"
        #warnings.warn("max mu pinned at max of mu search range", category=RuntimeWarning)
        warnings.warn(wm, category=RuntimeWarning)
    return (mu_min, mu_max)

def _round_to_nearest(value, nearest):
    output = round(value/nearest) * nearest
    return output

def FC_ints_search(n, b, alpha=0.1):
    """
    Like FC_ints_raw, but this tries to recursively find an approximate value for the bounds of
    the confidence interval.
    """
    assert isinstance(n, int) or isinstance(n,np.int64), "Input 'n' must be an integer"
    assert isinstance(b, float) or isinstance(b,np.float64), "Input 'b' must be a float"
    
    b_eff = max(b, 1e-10)
    fine_step = 0.005
    
    mu_best = max(0, n-b)
    mu_root_func = lambda mu: mu_acc(np.r_[mu], n, b_eff, alpha=alpha) - .5
    mu_upper_coarse = bisect(mu_root_func, mu_best, n+b_eff+10., xtol=1.)
    #mu_upper_coarse = _round_to_nearest(mu_upper_coarse, fine_step)
    mu_range_upper = np.arange(mu_upper_coarse-1.5, mu_upper_coarse+1.5,fine_step)
    mu_upper = mu_range_upper[mu_acc(mu_range_upper, n, b_eff, alpha=alpha)].max()
    if mu_upper == mu_range_upper.max():
        wm = f"did not find upper bound of mu: (n,b) = {(n,b)}"
        warnings.warn(wm, category=RuntimeWarning)
    
    mu_lower = 0.
    if mu_best > 1e-12:
        mu_left0, mu_right0 = 1e-12, mu_best # initial left and right windows of the bisect root search
        if mu_root_func(mu_left0) == mu_root_func(mu_right0):
            mu_lower = mu_left0
        else:
            mu_lower_coarse = bisect(mu_root_func, mu_left0, mu_right0, xtol=1.)
            #mu_lower_coarse = _round_to_nearest(mu_lower_coarse, fine_step)
            mu_lower_lower = max(1e-12, mu_lower_coarse-1.5)
            mu_lower_upper = mu_lower_coarse + 1.5
            mu_range_lower = np.arange(mu_lower_lower, mu_lower_upper, fine_step)
            cut_mu_acc = mu_acc(mu_range_lower, n, b_eff, alpha=alpha)
            mu_lower = mu_range_lower[mu_acc(mu_range_lower, n, b_eff, alpha=alpha)].min()
            if mu_lower == mu_lower_lower:
                wm = f"did not find lower bound of mu:\n (n,b) = {(n,b)}, {mu_lower = }, {mu_lower_coarse = }"
                warnings.warn(wm, category=RuntimeWarning)
    #return _round_to_nearest(mu_lower, fine_step), _round_to_nearest(mu_upper, fine_step)
    return mu_lower, mu_upper


def FC_ints(n, b, alpha=0.1):
    return FC_ints_raw(n, b, alpha=alpha)
FC_ints.__doc__ = FC_ints_raw.__doc__





















