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

def strat_test_brute(strata, sams, found, good, alternative='upper'):
    """
    Find p-value of the hypothesis that the number G of "good" objects in a 
    stratified population is less than or equal to good, using a stratified
    random sample.
    
    Assumes that a simple random sample of size sams[s] was drawn from stratum s, 
    which contains strata[s] objects in all.
    
    The P-value is the maximum Fisher combined P-value across strata
    over all allocations of good objects among the strata. The allocations are
    enumerated using Feller's "bars and stars" construction, constrained to honor the
    stratum sizes (each stratum can contain no more "good" items than it has items in all,
    nor fewer "good" items than the sample contains).
    
    The number of allocations grows combinatorially: there can be as many as
    [(#strata + #good items) choose (#strata-1)] allocations, making the brute-force
    approach computationally infeasible when the number of strata and/or the number of
    good items is large.
    
    The test is a union-intersection test: the null hypothesis is the union over allocations
    of the intersection across strata of the hypothesis that the number of good items
    in the stratum is less than or equal to a constant.
    
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
    best_part : list
        the partition that attained the maximum p-value
    """
    if alternative == 'upper': # exchange roles of "good" and "bad"
        p_func = lambda f, s, p, x: sp.stats.hypergeom.logcdf(f, s, p, x)
    elif alternative == 'lower':
        p_func = lambda f, s, p, x: sp.stats.hypergeom.logsf(f-1, s, p, x)
    else:
        raise NotImplementedError("alternative {} not implemented".format(alternative))
    best_part = found # start with what you see
    strata = np.array(strata, dtype=int)
    sams = np.array(sams, dtype=int)
    found = np.array(found, dtype=int)
    good = int(good)
    if good < np.sum(found):     
        p = 0 if alternative == 'lower' else 1 
    elif good > np.sum(strata) - np.sum(sams) + np.sum(found):
        p = 1 if alternative == 'lower' else 0
    else:  # use Feller's "bars and stars" enumeration of combinations, constrained
        log_p = np.NINF   # initial value for the max
        n_strata = len(strata)
        parts = sp.special.binom(good+n_strata-1, n_strata-1)
        if parts >= 10**7:
            print("warning--large number of partitions: {}".format(parts))
        barsNstars = good + n_strata
        bars = [0]*n_strata + [barsNstars]
        partition = ([bars[j+1] - bars[j] - 1 for j in range(n_strata)] \
            for bars[1:-1] in itertools.combinations(range(1, barsNstars), n_strata-1) \
            if all(((bars[j+1] - bars[j] - 1 <= strata[j]) and \
                   (bars[j+1] - bars[j] >= found[j])) for j in range(n_strata)))
        for part in partition:
            log_p_new = 0 
            for s in range(n_strata): # should be possible to vectorize this
                log_p_new += p_func(found[s], strata[s], part[s], sams[s])
            if log_p_new > log_p:
                best_part = part
                log_p = log_p_new
        p = sp.stats.chi2.sf(-2*log_p, df = 2*len(strata))
    return p, best_part

def strat_ci_bisect(strata, sams, found, alternative='lower', cl=0.95,
                  p_value=strat_test_brute):
    """
    Confidence bound on the number of ones in a stratified population,
    based on a stratified random sample (without replacement) from
    the population.
    
    
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
    alternative : string {'lower', 'upper'}
        if alternative=='lower', finds an upper confidence bound.
        if alternative=='upper', finds a lower confidence bound.
        While this is not mnemonic, it corresponds to the sidedness of the tests
        that are inverted to get the confidence bound.
    cl : float
        confidence level. Assumed to be at least 50%.
    p_value : callable
        method for computing the p-value
    
    Returns:
    --------
    b : int
        confidence bound
    best_part : list of ints
        partition that attains the confidence bound
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'upper':  # interchange good and bad
        compl = np.array(sams)-np.array(found)  # bad items found
        cb, best_part = strat_ci_bisect(strata, sams, compl, alternative='lower', 
                                       cl=cl, p_value=p_value)
        b = np.sum(strata) - cb    # good from bad
        best_part_b = np.array(strata, dtype=int)-np.array(best_part, dtype=int)
    else:
        cl_sidak = math.pow(cl, 1/len(strata))  # Šidák adjustment
        tail = 1-cl
        a = sum((hypergeom_conf_interval( \
                sams[s], found[s], strata[s], cl=cl_sidak, alternative="lower")[0] \
                for s in range(len(strata)))) # Šidák should give a lower bound
        b = int(np.sum(np.array(strata)*np.array(found)/np.array(sams)))-1 # expected good
        p_a, best_part_a = p_value(strata, sams, found, a, alternative=alternative)
        p_b, best_part_b = p_value(strata, sams, found, b, alternative=alternative)
        tot_found = np.sum(found)
        while p_a > tail and a > tot_found:
            a = math.floor(a/2)
            p_a, best_part_a = p_value(strata, sams, found, a, alternative=alternative)
        if p_a > tail:
            b = a
            best_part_b = best_part_a
        else:
            while b-a > 1:
                c = int((a+b)/2)
                p_c, best_part_c = p_value(strata, sams, found, 
                                           c, alternative=alternative)
                if p_c > tail:
                    b, p_b, best_part_b = c, p_c, best_part_c
                elif p_c < tail:
                    a, p_a, best_part_a = c, p_c, best_part_c
                elif p_c == tail:
                    b, p_b, best_part_b = c, p_c, best_part_c
                    break
    return b, list(best_part_b)
    
def strat_test(strata, sams, found, good, alternative='lower'):
    """
    P-value for the hypothesis the number of ones in a stratified population is not 
    greater than (or not less than) a hypothesized value, based on a stratified 
    random sample without replacement from the population.
    
    Uses a fast algorithm to find (an upper bound on) the P-value constructively.
    
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
    alternative : string {'lower', 'upper'}
        test against the alternative that the true number of "good" items 
        is less than (lower) or greater than (upper) the hypothesized number, good
    
    Returns:
    --------
    p : float
        P-value
    best_part : list
        the partition that attained the maximum p-value
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'upper':                 # exchange roles of "good" and "bad"
        compl = np.array(sams) - np.array(found) # bad items found 
        bad = np.sum(strata) - good            # total bad items hypothesized
        res = strat_test(strata, sams, compl, bad, alternative='lower')
        return res[0], list(np.array(strata, dtype=int)-np.array(res[1], dtype=int))
    else:  
        good = int(good)
        best_part = []               # best partition
        if good < np.sum(found):     # impossible
            p = 0  
            best_part = found
        elif good >= np.sum(strata) - np.sum(sams) + np.sum(found): # guaranteed
            p = 1
            best_part = list(np.array(strata, dtype=int) - \
                             np.array(sams, dtype=int) + \
                             np.array(found, dtype=int))       
        elif good >= np.sum(found):  # outcome is possible under the null  
            log_p = 0                # log of joint probability
            contrib = []             # contributions to the log joint probability
            base = np.sum(found)     # must have at least this many good items
            for s in range(len(strata)):
                log_p_j = sp.stats.hypergeom.logsf(found[s]-1, strata[s], found[s], sams[s])
                                     # baseline p for minimum number of good items in stratum
                log_p += log_p_j     # log of the product of stratum-wise P-values
                for j in range(found[s]+1, strata[s]-(sams[s]-found[s])+1):
                    log_p_j1 = sp.stats.hypergeom.logsf(found[s]-1, strata[s], j, sams[s])
                                     # tail probability for j good in stratum
                    contrib.append([log_p_j1 - log_p_j, s])  
                                     # relative increase in P from new item
                    log_p_j = log_p_j1
            sorted_contrib = sorted(contrib, key = lambda x: x[0], reverse=True)
            best_part = np.array(found)
            for i in range(good-base):
                log_p += sorted_contrib[i][0]
                best_part[int(sorted_contrib[i][1])] += 1
            p = sp.stats.chi2.sf(-2*log_p, df = 2*len(strata))
    return p, list(best_part)

def strat_ci_search(strata, sams, found, alternative='lower', cl=0.95):
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
        confidence level. Assumed to be at least 50%.
    
    Returns:
    --------
    cb : int
        confidence bound
    best_part : list of ints
        partition that attains the confidence bound (give or take one item)
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'upper':  # interchange good and bad
        compl = np.array(sams)-np.array(found)  # bad items found
        cb, best_part = strat_ci(strata, sams, compl, alternative='lower', cl=cl)
        cb = np.sum(strata) - cb    # good from bad
        best_part = np.array(strata, dtype=int)-np.array(best_part, dtype=int)
    else:
        cb = int(np.sum(np.array(strata)*np.array(found)/np.array(sams)))-1 # expected good
        p_attained, best_part = strat_test(strata, sams, found, cb, 
                                                 alternative=alternative)
        while p_attained >= 1-cl:
            cb -= 1
            p_attained, best_part = strat_test(strata, sams, found, cb, 
                                                     alternative=alternative)
        cb += 1
        p_attained, best_part = strat_test(strata, sams, found, cb, 
                                                 alternative=alternative)
    return cb, list(best_part)

def strat_ci(strata, sams, found, alternative='lower', cl=0.95):
    """
    Conservative confidence bound on the number of ones in a population,
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
    best_part : list of ints
        partition that attains the confidence bound (give or take one item)
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'upper':  # interchange role of good and bad
        compl = np.array(sams)-np.array(found)  # bad found
        cb, best_part = strat_ci(strata, sams, compl, alternative='lower', cl=cl)
        best_part = np.array(strata, dtype=int)-np.array(best_part, dtype=int)
        cb = np.sum(strata) - cb   # good to bad
    else:                
        threshold = -sp.stats.chi2.ppf(cl, df=2*len(strata))/2
        # g is in the set if 
        #   chi2.sf(-2*log(p), df=2*len(strata)) >= 1-cl
        #  i.e.,   -2*log(p) <=  chi2.ppf(cl, df)
        #  i.e.,   log(p) >= -chi2.ppf(cl, df)/2
        log_p = 0                # log of joint probability
        contrib = []             # contributions to the log joint probability
        base = np.sum(found)     # must have at least this many good items
        for s in range(len(strata)):
            log_p_j = sp.stats.hypergeom.logsf(found[s]-1, strata[s], found[s], sams[s])
                                 # baseline p for minimum number of good items in stratum
            log_p += log_p_j     # log of the product of stratum-wise P-values
            small = np.PINF      # for monotonicity check
            for j in range(found[s]+1, strata[s]-(sams[s]-found[s])+1):
                log_p_j1 = sp.stats.hypergeom.logsf(found[s]-1, strata[s], j, sams[s])
                log_p_j1 = log_p_j if log_p_j1 < log_p_j else log_p_j1 # true difference is nonnegative
                contrib.append([log_p_j1 - log_p_j, s]) 
                log_p_j = log_p_j1
                if contrib[-1][0] > small:
                    print("reversal in stratum {} for {} good; old: {} new:{}".format(s, 
                                    j, small, contrib[-1][0]))
                small = contrib[-1][0]
        sorted_contrib = sorted(contrib, key = lambda x: x[0], reverse=True)
        best_part = np.array(found)
        added = 0
        while log_p < threshold: 
            log_p += sorted_contrib[added][0]
            best_part[int(sorted_contrib[added][1])] += 1
            added += 1
        cb = base + added 
    return cb, list(best_part)

def strat_p(strata, sams, found, hypo, alternative='lower'):
    """
    Finds tail probability for the hypothesized population counts <hypo> for 
    simple random samples of sizes <sams> from strata of sizes <strata> if 
    <found> ones are found in the strata.
    
    Uses Fisher's combining function across strata.
    
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
        appropriate tail probability
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'lower':
        p_func = lambda x, N, G, n: sp.stats.hypergeom.sf(x-1, N, G, n)
    else:
        p_func = lambda x, N, G, n: sp.stats.hypergeom.cdf(x, N, G, n)
    p = 1    
    for s in range(len(strata)):
        p *= p_func(found[s], strata[s], hypo[s], sams[s])
    return sp.stats.chi2.sf(-2*math.log(p), df=2*len(strata))

def strat_p_ws(strata, sams, found, hypo, alternative='upper'):
    """
    Finds Wendell-Schmee P-value for the hypothesized population counts <hypo> for 
    simple random samples of sizes <sams> from strata of sizes <strata> if 
    <found> ones are found in the strata.
        
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
    assert alternative in ['lower', 'upper']
    if alternative == 'lower':                     # exchange roles of "good" and "bad"
        compl = np.array(sams) - np.array(found)   # bad items found 
        hypo_c = np.array(strata) - np.array(hypo) # total bad items hypothesized
        p = strat_p_ws(strata, sams, compl, hypo_c, alternative='upper')
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

def strat_test_ws(strata, sams, found, good, alternative='lower'):
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
    best_part : list
        the partition that attained the maximum p-value
    """
    assert alternative in ['lower', 'upper']
    if alternative == 'lower':                   # exchange roles of "good" and "bad"
        compl = np.array(sams) - np.array(found) # bad items found 
        bad = np.sum(strata) - good              # total bad items hypothesized
        res = strat_test_ws(strata, sams, compl, bad, alternative='upper')
        return res[0], list(np.array(strata, dtype=int)-np.array(res[1], dtype=int))        
    best_part = found # start with what you see
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
        if parts >= 10**7:
            print("warning--large number of partitions: {}".format(parts))
        barsNstars = good + n_strata
        bars = [0]*n_strata + [barsNstars]
        partition = ([bars[j+1] - bars[j] - 1 for j in range(n_strata)] \
            for bars[1:-1] in itertools.combinations(range(1, barsNstars), n_strata-1) \
            if all(((bars[j+1] - bars[j] - 1 <= strata[j]) and \
                   (bars[j+1] - bars[j] >= found[j])) for j in range(n_strata)))
        for part in partition:
            lo_t = (t for t in itertools.product(*[range(int(s+1)) for s in strat_max]) \
                    if p_hat(t) <= p_hat_0)
            p_new = sum(np.prod(sp.stats.hypergeom.pmf(t, strata, part, sams)) \
                    for t in lo_t)
            if p_new > p:
                best_part = part
                p = p_new
    return p, list(best_part)

def strat_ci_wright(strata, sams, found, alternative='lower', cl=0.95):
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
    assert alternative in ['lower', 'upper']
    inx = 0 if alternative == 'lower' else 1
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