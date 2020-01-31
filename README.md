# Efficient Exact Stratified Inference: EESI

Exact tests and confidence bounds for the population total of a binary population from a stratified simple random sample.

Wendell and Schmee (1996, https://www.tandfonline.com/doi/abs/10.1080/01621459.1996.10476950) proposed testing hypotheses about and finding confidence bounds for the population total by maximizing the $P$-value over a set of nuisance parameters---the individual stratum totals.
They find the $P$-value by ordering possible outcomes based on the estimated population total: 
Their approach is combinatorially complex: Feller's classic "bars and stars" argument shows that there are $\binom{G+S-1}{S-1}$ ways to allocate $G$ objects among $S$ strata.
(Some of those can be ruled out if $G$ exceeds the size of any stratum.)
Wendell and Schmee also provided R scripts for searching for the maximum over 
the allocations; the scripts became computationally impractical for more than three strata.

This document introduces a different strategy, also based on maximizing
the $P$-value over the nuisance parameters.
However, the $P$-value is based on the "raw" multivariate
hypergeometric counts, rather than on the estimated population total. 
A naive maximization of this $P$-value would also involve a search over
a combinatorial number of possible allocations.
However, no combinatorial search is necessary: the allocation that gives the largest
$P$-values and corresponding confidence bounds can be constructed in order $N \log N$ operations, where $N$ is the number of items in the population. The number $S$ of strata is immaterial.

The code herein implements both the brute-force approach (enumerate all ways of allocating a given number of ones across the strata, find the $P$-value for each, and find the maximum across all allocations) and the more efficient approach, which exploits special structure of the problem.