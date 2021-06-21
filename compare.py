import sys
!{sys.executable} -m pip install permute --user
!{sys.executable} -m pip install cryptorandom --user

import math
from scipy.stats import binom, hypergeom, chi2
import scipy as sp
import numpy as np
import itertools
import warnings
from permute.utils import binom_conf_interval, hypergeom_conf_interval

# New utility functions

def fisher_log(log_p, **kwargs):
    '''
    Fisher's combining function for the log of independent P-values
    
    Parameter
    ---------
    log_p : np.array or float
        vector of logarithms of independent P-values or sum of the logs
    df : int
        twice the number of log P-values in the sum. 
        Required if log_p is a scalar; otherwise, inferred from len(log_p)
        
    Returns
    -------
    P : float
        combined P-value (not on a log scale)
    '''
    if not isinstance(log_p, (list, tuple, np.ndarray)):  # log_p is already the sum; need sensible df
        df = kwargs.get('df',0)
        assert df >= 2, f'{df=} incorrect or not set'
    else: # there's a vector of log P-values; df is twice its length
        df = 2*len(log_p)
    return sp.stats.chi2.sf(-2*np.sum(log_p), df=df)

def bars_stars(strata, found, good):
    '''
    Generate all allocations of `good` 1s across the strata
    
    Parameters
    ----------
    strata : list of ints
        stratum sizes
    found : list of ints
        number of 1s found in the sampl from each stratum
    good : int
        number of 1s to distribute across the strata
    
    Returns
    -------
    generator that iterates over all allocations
    '''
    n_strata = len(strata)
    barsNstars = good + n_strata
    bars = [0]*n_strata + [barsNstars]
    return ([bars[j+1] - bars[j] - 1 for j in range(n_strata)]
            for bars[1:-1] in itertools.combinations(range(1, barsNstars), n_strata-1)
            if all(((bars[j+1] - bars[j] - 1 <= strata[j]) and \
            (bars[j+1] - bars[j] -1 >= found[j])) for j in range(n_strata)))            


class StratifiedBinary:
    '''
    allocation of 1s to strata

    Parameters
    ----------
    strata : numpy array of ints
        stratum sizes
    sams : numpy array of ints
        sample sizes
    found : numpy array of ints
        number of 1s in the sample from each stratum
    alloc : numpy array of ints
        initial allocation of 1s to strata (found)
    log_p : numpy array of floats
        tail probabilities for the allocation
    next_up : numpy array of floats
        log of the probability multipliers for including an additional 1 in each stratum
'''
    
    def __init__(self, strata=None, sams=None, found=None, alloc=None, log_p=None, next_up=None):
        self.strata = strata
        self.sams = sams
        self.found = found
        self.alloc = alloc
        self.log_p = log_p
        self.next_up = next_up
    
    def __str__(self):
        return f'{strata=}\n{sams=}\n{found=}\n{alloc=}\n{log_p=}\n{next_up=}'        
    
    def allocate_first(self):
        '''
        initialize the allocation of 1s to strata and ingredients for constructive max
        '''
        self.alloc = self.found.copy()                    # allocation that maximizes the P-value so far
        self.log_p = np.array([sp.stats.hypergeom.logsf(self.found[s]-1, self.strata[s], self.alloc[s], \
                             self.sams[s]) for s in range(len(self.strata))])  # stratumwise log tail probabilities
        self.next_up = np.array([np.NINF if self.alloc[s]+1 > self.strata[s]-self.sams[s]+self.found[s] \
                                 else sp.stats.hypergeom.logsf(self.found[s]-1, self.strata[s], self.alloc[s]+1,\
                                self.sams[s]) - self.log_p[s] for s in range(len(self.strata))])
        return True

    def allocate_next(self):
        '''
        allocate an additional 1 to the stratum that gives largest tail probability

        updates alloc and next_up in place
        '''
        big = np.argmax(self.next_up)
        self.alloc[big] += 1
        self.log_p[big] = sp.stats.hypergeom.logsf(self.found[big]-1, self.strata[big], self.alloc[big], self.sams[big])
        self.next_up[big] = (np.NINF if self.alloc[big]+1 > self.strata[big]-self.sams[big]+self.found[big]
                             else sp.stats.hypergeom.logsf(self.found[big]-1, self.strata[big], self.alloc[big]+1, 
                             self.sams[big]) - self.log_p[big])
        return (True if np.max(self.next_up) > np.NINF else False)
    
    def fisher_p(self):
        '''
        Fisher P-value
        '''
        return fisher_log(self.log_p)
    
    def total(self):
        '''
        total 1s allocated
        '''
        return np.sum(self.alloc)
    
    def n_strata(self):
        '''
        number of strata
        '''
        return len(self.strata)
    
# Maximum P-values over allocations 
            
def strat_test_brute(strata, sams, found, good, **kwargs):
    '''
    p-value of the hypothesis that the number of 1s in a binary population is 
    less than or equal to `good`, from a stratified random sample.
    
    Assumes that a simple random sample of size sams[s] was drawn from stratum s, 
    which contains strata[s] objects in all.
    
    The P-value is the maximum Fisher combined P-value across strata
    over all allocations of good 1s among the strata. The allocations are
    enumerated using Feller's "bars and stars" construction, constrained to honor the
    stratum sizes and the data (each stratum can contain no more 1s than it has items in all
    minus the observed number of 0s, nor fewer "good" items than the sample contains).
    
    The number of allocations grows combinatorially: there can be as many as
    [(#strata + #1s) choose (#strata-1)] allocations, making the brute-force approach computationally 
    infeasible when the number of strata and/or the number of 1s is large.
    
    The test is a union-intersection test: the null hypothesis is the union over allocations
    of the intersection across strata of the hypothesis that the number of 1s
    in the stratum is less than or equal to a constant.
    
    Parameters:
    -----------
    strata : list of ints
        sizes of the strata
    sams : list of ints
        sample sizes from the strata
    found : list of ints
        the numbers of 1s found in the samples from the strata
    good : int
        the hypothesized total number of 1s in the population
    kwargs : keyword arguments for this function and the functions it calls
        alternative : string {'lower', 'upper'}
            test against the alternative that the true number of 1s is less than good (lower)
            or greater than good ('upper'). Default 'lower'
        combining_function : callable 
            combining function; default is fisher_log. 
            kwarg is also passed to combining_function
        cheap_combiner : callable
            monotone increasing function of the combining function. 
            Default np.sum if combining_function == fisher_log
            kwarg also passed to cheap_combiner
        warn : int
            warn if the number of allocations exceeds this. Default 10**7        
    
    Returns:
    --------
    p : float
        maximum combined p-value over all ways of allocating good "good" objects
        among the strata, honoring the stratum sizes.        
    alloc : list
        an allocation that attains the maximum p-value
    '''
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower','upper'], f'alternative {alternative} not implemented'
    alloc = None
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    strata = np.array(strata, dtype=int)
    if good < np.sum(found):     
        p = 0 if alternative == 'lower' else 1 
    elif good > np.sum(strata) - np.sum(sams) + np.sum(found):
        p = 1 if alternative == 'lower' else 0
    else:  
        if alternative == 'upper':                   # exchange roles of 1s and 0s
            compl = sams - found                     # 0s found 
            bad = np.sum(strata) - good              # total 0s hypothesized
            kwargs_c = kwargs.copy()
            kwargs_c['alternative'] = 'lower'
            p, alloc_c = strat_test_brute(strata, sams, compl, bad, **kwargs_c)
            alloc = None if alloc_c is None else list(strata-np.array(alloc_c, dtype=int))
        else:
            p = np.NINF   # initial value for the max
            n_strata = len(strata)
            parts = sp.special.binom(good+n_strata-1, n_strata-1)
            combining_function = kwargs.get('combining_function', fisher_log)
            if combining_function == fisher_log:
                kwargs['df'] = 2*n_strata
                cheap_combiner = lambda p_vec, **kwargs: np.sum(p_vec)
            else:
                cheap_combiner = kwargs.get('cheap_combiner', combining_function)
            warn = kwargs.get('warn',10**7)
            if parts >= warn:
                print(f'warning--large number of allocations: {parts}')
            alloc = found.copy()
            for part in bars_stars(strata, found, good):
                p_new = cheap_combiner( 
                            np.array([sp.stats.hypergeom.logsf(found[s]-1, strata[s], part[s], sams[s])
                            for s in range(n_strata)]),
                            **kwargs)
                if p_new > p:
                    alloc = part
                    p = p_new
            p = combining_function(p, **kwargs)
    return p, (None if alloc is None else list(alloc))
    
def strat_test(strata, sams, found, good, **kwargs):
    """
    P-value for the hypothesis that the number of 1s in a binary population is not 
    greater than (or not less than) a hypothesized value, based on a stratified 
    random sample without replacement.
    
    Uses the fast algorithm to find the P-value constructively.
    
    Uses Fisher's combining function to combine stratum-level P-values.
    
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes in the strata
    found : list of ints
        number of ones found in each stratum in each sample
    good : int
        hypothesized number of ones in the population
    kwargs : dict
        alternative : string {'lower', 'upper'} default 'lower'
            test against the alternative that the true number of 1s is less than (lower) 
            or greater than (upper) the hypothesized number, good
    
    Returns:
    --------
    p : float
        P-value
    alloc : list
        an allocation that attains the maximum p-value
    """
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower', 'upper'], f'alternative {alternative} not implemented'
    strata = np.array(strata, dtype=int)
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    good = int(good)
    alloc = None
    if good < np.sum(found):     
        p = 0 if alternative == 'lower' else 1 
    elif good > np.sum(strata) - np.sum(sams) + np.sum(found):
        p = 1 if alternative == 'lower' else 0
    else:
        if alternative == 'upper':                  # exchange roles of "good" and "bad"
            compl = sams - found                    # bad items found 
            bad = np.sum(strata) - good             # total bad items hypothesized
            kwargs_c = kwargs.copy()
            kwargs_c['alternative'] = 'lower'
            p, alloc_c = strat_test(strata, sams, compl, bad, **kwargs_c)
            alloc = (None if alloc_c is None 
                          else list(strata-np.array(alloc_c, dtype=int)))
        else:  
            if good < np.sum(found) or good > np.sum(strata - sams + found): # impossible
                p = 0  
            elif good == np.sum(strata-sams+found): # the "packed" allocation guarantees this outcome or more 1s
                p = 1
                alloc = strata-sams+found      
            else:                                   # outcome is possible but not certain under the composite null 
                optimal = StratifiedBinary(strata=strata, sams=sams, found=found)
                optimal.allocate_first()
                while optimal.total() < good:
                    optimal.allocate_next()
                p = optimal.fisher_p()
                alloc = optimal.alloc
    return p, (None if alloc is None else list(alloc))

# Confidence intervals

def strat_ci_bisect(strata, sams, found, **kwargs):
    """
    Confidence bound on the number of ones in a stratified binary population,
    based on a stratified random sample without replacement
    
    If alternative=='lower', finds an upper confidence bound.
    If alternative=='upper', finds a lower confidence bound.

    Uses an integer bisection search to find an exact confidence bound.
    The starting upper endpoint for the search is the unbiased estimate
    of the number of ones in the population. That could be refined in various
    ways to improve efficiency.
    
    The lower endpoint for the search is the Šidák joint lower confidence bounds,
    which should be more conservative than the exact bound.
    
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes in the strata
    found : list of ints
        number of ones found in each stratum in each sample
    kwargs:
        alternative : string in {'lower', 'upper'}
            if alternative=='lower', finds an upper confidence bound.
            if alternative=='upper', finds a lower confidence bound.
            While this is not mnemonic, it corresponds to the sidedness of the tests
            that are inverted to get the confidence bound.
        cl : float
            confidence level. Assumed to be at least 0.5. Default 0.95.
        p_value : callable
            method for computing the p-value
        kwargs is also passed to p_value
    
    Returns:
    --------
    b : int
        confidence bound
    alloc : list of ints
        allocation that attains the confidence bound
    """
    cl = kwargs.get('cl',0.95)
    p_value = kwargs.get('p_value', strat_test_brute)
    alternative = kwargs.get('alternative','lower')
    strata = np.array(strata, dtype=int)
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    assert alternative in ['lower', 'upper'], f'alternative {alternative} not implemented'    
    if alternative == 'upper':  # interchange good and bad
        compl = sams-found      # bad items found
        kwargs_c = kwargs.copy()
        kwargs_c['alternative'] = 'lower'
        cb, alloc_c = strat_ci_bisect(strata, sams, compl, **kwargs_c)
        b = np.sum(strata) - cb    # good from bad
        alloc = strata-np.array(alloc_c, dtype=int)
    else:
        cl_sidak = math.pow(cl, 1/len(strata))  # Šidák adjustment
        tail = 1-cl
        a = sum((hypergeom_conf_interval( \
                sams[s], found[s], strata[s], cl=cl_sidak, alternative="lower")[0] \
                for s in range(len(strata)))) # Šidák should give a lower bound
        b = int(np.sum(np.array(strata)*np.array(found)/np.array(sams)))-1 # expected good
        p_a, alloc_a = p_value(strata, sams, found, a, alternative=alternative)
        p_b, alloc = p_value(strata, sams, found, b, alternative=alternative)
        tot_found = np.sum(found)
        while p_a > tail and a > tot_found:
            a = math.floor(a/2)
            p_a, alloc_a = p_value(strata, sams, found, a, **kwargs)
        if p_a > tail:
            b = a
            alloc = alloc_a
        else:
            while b-a > 1:
                c = int((a+b)/2)
                p_c, alloc_c = p_value(strata, sams, found, c, **kwargs)
                if p_c > tail:
                    b, p_b, alloc = c, p_c, alloc_c
                elif p_c < tail:
                    a, p_a, alloc_a = c, p_c, alloc_c
                elif p_c == tail:
                    b, p_b, alloc = c, p_c, alloc_c
                    break
    return b, list(alloc)

def strat_ci_search(strata, sams, found, **kwargs):
    """
    Confidence bound on the number of ones in a stratified population,
    based on a stratified random sample (without replacement) from
    the population.
        
    If alternative=='lower', finds an upper confidence bound.
    If alternative=='upper', finds a lower confidence bound.
    
    Searches for the allocation of items that attains the confidence bound
    by increasing the number of ones from the minimum consistent
    with the data (total found in the sample) until the P-value is greater
    than 1-cl.
    
    Uses the fast method for finding the maximum P-value for Fisher's combining function
    
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes in the strata
    found : list of ints
        number of ones found in each stratum in each sample
    kwargs : dict
        alternative : string {'lower', 'upper'} Default 'lower'
            if alternative=='lower', finds an upper confidence bound.
            if alternative=='upper', finds a lower confidence bound.
            While this is not mnemonic, it corresponds to the sidedness of the tests
            that are inverted to get the confidence bound.
        cl : float Default 0.95
            confidence level. Assumed to be at least 50%.
    
    Returns:
    --------
    cb : int
        confidence bound
    alloc : list of ints
        allocation that attains the confidence bound (give or take one item)
    """
    cl = kwargs.get('cl',0.95)
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower', 'upper'], f'alternative {alternative} not implemented'
    strata = np.array(strata, dtype=int)
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    if alternative == 'upper':  # interchange good and bad
        kwargs_c = kwargs.copy()
        kwargs_c['alternative'] = 'lower'
        compl = sams-found  # bad items found
        cb, alloc_c = strat_ci(strata, sams, compl, **kwargs_c)
        cb = np.sum(strata) - cb    # good from bad
        alloc = strata - alloc_c
    else:
        cb = int(np.sum(strata*found/sams))-1 # expected good
        p_attained, alloc = strat_test(strata, sams, found, cb, alternative=alternative)
        while p_attained >= 1-cl:
            cb -= 1
            p_attained, alloc = strat_test(strata, sams, found, cb, alternative=alternative)
        cb += 1
        p_attained, alloc = strat_test(strata, sams, found, cb, alternative=alternative)
    return cb, list(alloc)

def strat_ci(strata, sams, found, **kwargs):
    """
    Confidence bound on the number of ones in a population,
    based on a stratified random sample (without replacement) from
    the population.
    
    If alternative=='lower', finds an upper confidence bound.
    If alternative=='upper', finds a lower confidence bound.
    
    Constructs the confidence bound directly by constructing the
    allocation of the maximum number of ones that would not be
    rejected at (conservative) level 1-cl.
    
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes in the strata
    found : list of ints
        number of ones found in each stratum in each sample
    kwargs : dict
        alternative : string {'lower', 'upper'} default 'lower'
            if alternative=='lower', finds an upper confidence bound.
            if alternative=='upper', finds a lower confidence bound.
            While this is not mnemonic, it corresponds to the sidedness of the tests
            that are inverted to get the confidence bound.
        cl : float default 0.95
            confidence level

    Returns:
    --------
    cb : int
        confidence bound
    alloc : list of ints
        allocation that attains the confidence bound (give or take one item)
    """
    cl = kwargs.get('cl',0.95)
    alternative = kwargs.get('alternative','lower')
    strata = np.array(strata, dtype=int)
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    assert alternative in ['lower', 'upper'], f'alternative {alternative} not implemented'
    if alternative == 'upper':  # interchange role of good and bad
        compl = sams - found  # bad found
        kwargs_c = kwargs.copy()
        kwargs_c['alternative'] = 'lower'
        cb, alloc = strat_ci(strata, sams, compl, **kwargs_c)
        alloc = strata - alloc
    else:                
        threshold = -sp.stats.chi2.ppf(cl, df=2*len(strata))/2
        # g is in the confidence set if 
        #          chi2.sf(-2*log(p), df=2*len(strata)) >= 1-cl
        #  i.e.,   -2*log(p) <=  chi2.ppf(cl, df)
        #  i.e.,   log(p) >= -chi2.ppf(cl, df)/2
        optimal = StratifiedBinary(strata=strata, sams=sams, found=found)
        optimal.allocate_first()
        while np.sum(optimal.log_p) < threshold:
            optimal.allocate_next()
        alloc = optimal.alloc
    return np.sum(alloc), list(alloc)

# older methods

def strat_p_ws(strata, sams, found, hypo, **kwargs):
    """
    Finds Wendell-Schmee P-value for the hypothesized population counts 'hypo' for 
    simple random samples of sizes 'sams' from strata of sizes 'strata' if 
    'found' 1s are found in the samples from the strata.
        
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes from the strata
    found : list of ints
        number of ones found in each stratum in each sample
    hypo : list of ints
        hypothesized number of ones in the strata
    
    Returns:
    --------
    p : float
        tail probability
    """
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower', 'upper']
    if alternative == 'lower':                     # exchange roles of "good" and "bad"
        kwargs_c = kwargs.copy()
        kwargs_c['alternative'] = 'upper'
        compl = np.array(sams) - np.array(found)   # bad items found 
        hypo_c = np.array(strata) - np.array(hypo) # total bad items hypothesized
        p = strat_p_ws(strata, sams, compl, hypo_c, **kwargs_c)
    else:    
        p_hat = lambda f, st=strata, sa=sams: np.sum(np.array(st)*np.array(f)/np.array(sa))/np.sum(st) # pooled estimate
        p_hat_0 = p_hat(found)
        per_strat = np.array(strata)/np.array(sams)/np.sum(strata)
        strat_max = np.floor(p_hat_0/per_strat)
        lo_t = (t for t in itertools.product(*[range(int(s+1)) for s in strat_max]) \
                    if p_hat(t) <= p_hat_0)
        p = sum(np.prod(sp.stats.hypergeom.pmf(t, strata, hypo, sams)) \
                    for t in lo_t)
    return p

def strat_test_ws(strata, sams, found, good, **kwargs):
    """
    Find p-value of the hypothesis that the number G of "good" objects in a 
    stratified population is less than or equal to good, using a stratified
    random sample.
    
    Assumes that a simple random sample of size sams[s] was drawn from stratum s, 
    which contains strata[s] objects in all.
    
    The P-value is the maximum Windell-Schmee P-value over all allocations of 
    good objects among the strata. The allocations are enumerated using Feller's 
    "bars and stars" construction, constrained to honor the stratum sizes (each 
    stratum can contain no more "good" items than it has items in all, nor fewer 
    "good" items than the sample contains).
    
    The number of allocations grows combinatorially: there can be as many as
    [(#strata + #good items) choose (#strata-1)] allocations, making the brute-force
    approach computationally infeasible when the number of strata and/or the number of
    good items is large.
    
    Parameters:
    -----------
    strata : list of ints
        sizes of the strata. One int per stratum.
    sams : list of ints
        the sample sizes from each stratum
    found : list of ints
        the numbers of "good" items found in the samples from the strata
    good : int
        the hypothesized total number of "good" objects in the population
    alternative : string {'lower', 'upper'}
        test against the alternative that the true value is less than good (lower)
        or greater than good (upper)
    
    Returns:
    --------
    p : float
        maximum combined p-value over all ways of allocating good "good" objects
        among the strata, honoring the stratum sizes.        
    alloc : list
        the allocation that attained the maximum p-value
    """
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower', 'upper']
    if alternative == 'lower':                   # exchange roles of "good" and "bad"
        compl = np.array(sams) - np.array(found) # bad items found 
        bad = np.sum(strata) - good              # total bad items hypothesized
        kwargs_c = kwargs.copy()
        kwargs_c['alternative'] = 'upper'
        res = strat_test_ws(strata, sams, compl, bad, **kwargs_c)
        return res[0], list(np.array(strata, dtype=int)-np.array(res[1], dtype=int))        
    alloc = found # start with what you see
    good = int(good)
    if good < np.sum(found):     
        p = 0 if alternative == 'lower' else 1 
    elif good > np.sum(strata) - np.sum(sams) + np.sum(found):
        p = 1 if alternative == 'lower' else 0
    else:  # use Feller's "bars and stars" enumeration of combinations, constrained
        p_hat = lambda f, st=strata, sa=sams: np.sum(np.array(st)*np.array(f)/np.array(sa))/np.sum(st) # pooled estimate
        p_hat_0 = p_hat(found)
        per_strat = np.array(strata)/np.array(sams)/np.sum(strata)
        strat_max = np.floor(p_hat_0/per_strat)
        p = 0   # initial value for the max
        n_strata = len(strata)
        parts = sp.special.binom(good+n_strata-1, n_strata-1)
        for part in bars_stars(strata, found, good):
            lo_t = (t for t in itertools.product(*[range(int(s+1)) for s in strat_max]) \
                    if p_hat(t) <= p_hat_0)
            p_new = 0
            for t in lo_t:
                p_temp = 1
                for s in range(len(strata)):
                    p_temp *= sp.stats.hypergeom.pmf(t[s], strata[s], part[s], sams[s])
                p_new += p_temp
            if p_new > p:
                alloc = part
                p = p_new
    return p, list(alloc)

def strat_ci_wright(strata, sams, found, **kwargs):
    """
    Confidence bound on the number of ones in a stratified population,
    based on a stratified random sample (without replacement) from
    the population.
    
    If alternative=='lower', finds an upper confidence bound.
    If alternative=='upper', finds a lower confidence bound.
    
    Constructs the confidence bound by finding Šidák multiplicity-adjusted
    joint lower confidence bounds for the number of ones in each stratum.
    
    This approach is mentioned in Wright, 1991.
    
    Parameters:
    -----------    
    strata : list of ints
        stratum sizes
    sams : list of ints
        sample sizes in the strata
    found : list of ints
        number of ones found in each stratum in each sample
    alternative : string {'lower', 'upper'}
        if alternative=='lower', finds an upper confidence bound.
        if alternative=='upper', finds a lower confidence bound.
        While this is not mnemonic, it corresponds to the sidedness of the tests
        that are inverted to get the confidence bound.
    cl : float
        confidence level
        
    Returns:
    --------
    cb : int
        confidence bound
    """
    alternative = kwargs.get('alternative','lower')
    assert alternative in ['lower', 'upper']
    inx = 0 if alternative == 'lower' else 1
    cl = kwargs.get('cl',0.95)
    cl_sidak = math.pow(cl, 1/len(strata))  # Šidák-adjusted confidence level per stratum
    cb = sum((hypergeom_conf_interval(
                sams[s], found[s], strata[s], cl=cl_sidak, alternative=alternative)[inx] 
                for s in range(len(strata))))
    return cb
    
# When is new sharper than WS?
strata = [100, 100, 100, 100]
sams = [25,25,25,25]
found = [20,0,0,0] 
alternative = 'upper'
print("strata: {} sams: {} found: {}".format(strata, sams, found))
print(\
      strat_ci_bisect(strata, sams, found, alternative=alternative, p_value=strat_test_ws),\
      strat_ci(strata, sams, found, alternative=alternative),\
      strat_ci_wright(strata, sams, found, alternative=alternative))