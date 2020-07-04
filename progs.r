# THIS R CODE IS ADAPTED FROM THE S-PLUS Code WENDELL AND SCHMEE (1996) have developed.
#
#
#

# You take a stratified sample of sizes (75,50,25) from a population divided into
# three strata of sizes (500,300,200) and find errors in each sample of (2,1,0).
# A p-value on the hypothesis that the population contains 50 errors against the
# alternate that the population contains fewer than 50 errors can be found using the
# pmax.function as follows:
#
# > pmax.function(c(500,300,200),c(75,50,25),c(2,1,0),50)
#            [,1] [,2] [,3] [,4]
# [1,] 0.02767569   28   21    1
#
# indicating a maximum p-value of 0.02768 occurring under the null of 50 errors
# with those errors distributed as (28,21,1).
#
# A 95% confidence interval on the number of errors in the population can
# be found using interval.strat as follows:
# > interval.strat(c(500,300,200),c(75,50,25),c(2,1,0),.95)
# [1]  8 50
#
# THE FUNCTIONS:
#
# The functions are either what I call top level functions, functions that evaluate
# a stratified sample; subroutine functions, functions that are used by the top
# level functions but have no real purpose as a stand alone function; or utilities,
# functions that are used by the top level functions like a subroutine but also
# have a purpose as a stand alone function.
#
# TOP LEVEL FUNCTIONS:
#
pmax.function <- function(N, n, y, Th, width = 0.2)
# This function calculates a p-value for a stratified sample
# from a finite population where an item in the population
# is either in error or not in error.  The p-value is evaluated
# against the null number of errors, Th, with the alternate
# that the errors in the population are less than Th.
#
# N is a vector of the population sizes for each strata.
# n is a vector of the sample sizes for each strata.
# y is a vector of the number of errors in each sample for each strata.
# Th is the the number of errors in the population under the null hypothesis.
# If length(N) > 5 and/or sum(y) > 10 very long computational times may be encountered.
#
# This function returns a vector with a maximum p-value (pmax) in the
# first position and then the position for where pmax was found in the
# parameter space in the next L positions.
# If multiple maximums were found then a matrix is returned with each row
# a vector identifying a maximum.
#
# The function finds the maximum p-value for the parameter space defined
# by the null hypothesis using gradient searches starting from each corner
# point in the parameter space.
#
# width sets the intial width of the search in terms of the strata sizes.
# I have found the default of .2 to work well, but in critical applications
# it would be prudent to try several different widths to be sure the true
# maximum was not missed although in extensive testing the true maximum was
# never missed with width = .2
{
        ym <- ym.function(N, n, y)
        nym <- newym(ym)
        permutations <- permute.function(length(N))
        n.matrix <- n.matrix.function(permutations)
        search.function <- function(N, M, n, y, ym, nym, n.matrix, width)
        {
                search1.function <- function(N, M, n, y, ym, nym, n.matrix, width)
                {
                        neighborhood.function <- function(N, M, n, y, n.matrix, width)
                        {
                                x <- t(t(ifelse(n.matrix <= 2, (3 - 2 * n.matrix) * width, 0)) + M)
                                x <- rbind(x[apply(x, 1, function(N, n, y, x)
                                all(x >= y & x <= N - n + y), N = N, n = n, y = y),  ], M)
                                dimnames(x) <- NULL
                                x
                        }
                        pmax1 <- 1
                        pmax2 <- 2
                        while(pmax1 != pmax2) {
                                pmax2 <- probfunct(N, M, n, ym, nym)
                                x <- neighborhood.function(N, M, n, y, n.matrix, width)
                                z <- apply(x, 1, probfunct, N = N, n = n, ym = ym, nym = nym)
                                pmax1 <- max(z)
                                M <- x[z == max(z),  ]
                                if(length(M) != length(N)) {
                                  M <- M[1,  ]
                                }
                        }
                        c(pmax1, M)
                }
                width <- max(trunc(width * sum(M)), 2)
                x <- c(0, M)
                while(width > 1) {
                        width <- trunc(width/2)
                        M <- x[-1]
                        x <- search1.function(N, M, n, y, ym, nym, n.matrix, width)
                }
                x
        }
        x <- corner.function(N, n, y, Th, permutations = permutations)
        x <- t(apply(x, 1, search.function, N = N, n = n, y = y, ym = ym, nym = nym, n.matrix = n.matrix, width = width))
        x <- x[x[, 1] == max(x[, 1]),  ]
        unique.rows(x)
}
#
ubstrat.function <- function(N, n, y, cl = 0.95)
# see pmax.function for explaination of N,n,y.
# cl is the confidence level.
# This function returns a vector with the upper confidence limit
# in the first position, pmax at that upper confidence limit in the second
# position and then the coordinates where the pmax occurred in the next L positions.
# The bound is found by inverting the p-value, pmax, using the method described
# by Buonaccorsi, J. (1987), "A Note on Confidence Intervals for Proportions
# in Finite Populations," The American Statistician, 41, 215 - 218.
{
        alpha <- 1 - cl
        ym <- ym.function(N, n, y)
        nym <- newym(ym)
        permutations <- permute.function(length(N))
        n.matrix <- n.matrix.function(permutations)
        Th <- ubnest.function(N, n, y, ym, nym, permutations, n.matrix, cl)
        x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        while(x[1] > alpha) {
                Th <- Th + 1
                x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        }
        while(x[1] <= alpha) {
                Th <- Th - 1
                x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        }
        c(Th, x)
}
#
lbstrat.function <- function(N, n, y, cl = 0.95)
# works just like ubstrat.function except it calculates a lower bound.
{
        alpha <- 1 - cl
        ym <- ym.prime(N, n, y)
        nym <- newym(ym)
        permutations <- permute.function(length(N))
        n.matrix <- n.matrix.function(permutations)
        Th <- lbnest.function(N, n, y, ym, nym, permutations, n.matrix, cl)
        x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        while(1 - x[1] > alpha) {
                Th <- Th - 1
                x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        }
        while(1 - x[1] <= alpha) {
                Th <- Th + 1
                x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        }
        x[1] <- 1 - x[1]
        c(Th, x)
}
#
interval.strat <- function(N, n, y, cl = 0.95)
# uses ubstrat.function and lbstrat.function to construct
# a confidence interval.
{
        cl <- 1 - (1 - cl)/2
        ub <- ubstrat.function(N, n, y, cl)
        lb <- lbstrat.function(N, n, y, cl)
        c(lb[1], ub[1])
}
#
# UTILITIES:
#
hypergeo <- function(N, M, n, y)
# calculates the probability of observing y errors in a sample
# of size n from a population with M errors of size N.
{
        dhyper(y, M, N - M, n)
}
#
chypergeo <- function(N, M, n, y)
# calculates the probability of observing y  or fewer errors in
# a sample of size n from a population with M errors of size N.
{
        phyper(y, M, N - M, n)
}
#
ubbinomial <- function(n, y, cl)
# calculates an upper confidence bound at confidence level cl on the
# error rate in an infinite population given a simple random sample
# of size n containing y errors.
{
        1/(1 + (n - y)/((y + 1) * qf(cl, 2 * y + 2, 2 * n - 2 * y)))
}
#
lbbinomial <- function(n, y, cl)
# calculates a lower confidence bound at confidence level cl on the
# error rate in an infinite population given a simple random sample
# of size n containing y errors.
{
        1/(1 + ((n - y + 1) * qf(cl, 2 * n - 2 * y + 2, 2 * y))/y)
}
#
 ubhyper <- function(N, n, y, cl)
# calculates an upper confidence bound at confidence level cl on the
# number of errors in a population of size N given a simple random sample
# of size n containing y errors.
# The bound is found by inverting the p-value using the method described
# by Buonaccorsi, J. (1987), "A Note on Confidence Intervals for Proportions
# in Finite Populations," The American Statistician, 41, 215 - 218.
{
        ubhyper1 <- function(N, n, y, cl)
        {
                alpha <- 1 - cl
                M <- round(((ubbinomial(n, y, cl) - y/n) * sqrt((N - n)/N) + y/n) * N)
                p <- chypergeo(N, M, n, y)
                while(p > alpha) {
                        M <- M + 1
                        p <- chypergeo(N, M, n, y)
                }
                while(p <= alpha) {
                        M <- M - 1
                        p <- chypergeo(N, M, n, y)
                }
                M
        }
        x <- c(1:length(N))
        for(i in 1:length(N)) {
                x[i] <- ubhyper1(N[i], n[i], y[i], cl)
        }
        x
}
#
 lbhyper <- function(N, n, y, cl)
# calculates a lower confidence bound at confidence level cl on the
# number of errors in a population of size N given a simple random sample
# of size n containing y errors.
# The bound is found by inverting the p-value using the method described
# by Buonaccorsi, J. (1987), "A Note on Confidence Intervals for Proportions
# in Finite Populations," The American Statistician, 41, 215 - 218.
{
        lbhyper1 <- function(N, n, y, cl)
        {
                if(y == 0)
                        M <- 0
                else {
                        alpha <- 1 - cl
                        M <- round((y/n - (y/n - lbbinomial(n, y, cl)) * sqrt((N - n)/N)) * N)
                        p <- 1 - chypergeo(N, M, n, y - 1)
                        while(p > alpha) {
                                M <- M - 1
                                p <- 1 - chypergeo(N, M, n, y - 1)
                        }
                        while(p <= alpha) {
                                M <- M + 1
                                p <- 1 - chypergeo(N, M, n, y - 1)
                        }
                }
                M
        }
        x <- c(1:length(N))
        for(i in 1:length(N)) {
                x[i] <- lbhyper1(N[i], n[i], y[i], cl)
        }
        x
}
#
phat.function <- function(N, n, y)
# calculates and unbiased point estimate of the number of errors in
# a population of size N, given sample of size n containing y errors.
{
        sum(y/n * N)
}
#
combinations.function <- function(levels)
# levels is a vector of integers >= 1.  This function returns a matrix of
# all possible combinations for the numbers 1...levels.
{
        nrows <- prod(levels)
        ncols <- length(levels)
        yy <- matrix(0, nrows, ncols)
        rep1 <- 1
        for(i in 1:ncols) {
                lev <- 1:levels[i]
                yy[, i] <- rep(rep(lev, rep(rep1, levels[i])), length = nrows)
                rep1 <- rep1 * levels[i]
        }
        yy
}
#
 permute.function <- function(n)
# returns a matrix of all possible permutations for the
# integers 1 ... n.
{
        perm <- function(J, n)
        {
# return the J'th permutation of the integers 1:n,
# J should be an integer in range 1:factorial(n).
# Written by Bob Henery, 1992, Toulouse.
                if(n == 0) return(integer(0))
                if(n == 1)
                        return(as.integer(1))
                J <- J - 1
                per <- 1
                for(ind in 2:n) {
                        nleft <- J %% ind
                        J <- J %/% ind
                        if(nleft == 0)
                                per <- c(ind, per)
                        else {
                                if(nleft == ind - 1)
                                  per <- c(per, ind)
                                else per <- c(per[1:nleft], ind, per[(1 + nleft):(ind - 1)])
                        }
                }
                per
        }
        t(sapply(seq(gamma(n + 1)), perm, n = n))
}
#
unique.rows <- function(x)
# x is a matrix. The function returns a matrix that
# contains the unique rows of x.
{
        matrix.sort.byrow <- function(x)
        {
                for(i in seq(ncol(x), 1, -1)) {
                        x <- x[order(x[, i]),  ]
                }
                x
        }
        matrix.row.cut <- function(x)
        {
                a <- rep(T, nrow(x))
                for(i in 2:nrow(x)) {
                        a[i] <- any(x[i - 1,  ] != x[i,  ])
                }
                x[a,  , drop = F]
        }
        matrix.row.cut(matrix.sort.byrow(x))
}
#
# SUBROUTINES:
#
pmax.short <- function(N, Th, n, y, ym, nym, permutations, n.matrix, width = 0.2)
# version of pmax.function used by ubstrat.function and lbstrat.function.
# essentially the same as pmax.function except it does not calculate ym, nym, or
# n.matrix.
{
        search.function <- function(N, M, n, y, ym, nym, n.matrix, width)
        {
                search1.function <- function(N, M, n, y, ym, nym, n.matrix, width)
                {
                        neighborhood.function <- function(N, M, n, y, n.matrix, width)
                        {
                                x <- t(t(ifelse(n.matrix <= 2, (3 - 2 * n.matrix) * width, 0)) + M)
                                x <- rbind(x[apply(x, 1, function(N, n, y, x)
                                all(x >= y & x <= N - n + y), N = N, n = n, y = y),  ], M)
                                dimnames(x) <- NULL
                                x
                        }
                        pmax1 <- 1
                        pmax2 <- 2
                        while(pmax1 != pmax2) {
                                pmax2 <- probfunct(N, M, n, ym, nym)
                                x <- neighborhood.function(N, M, n, y, n.matrix, width)
                                z <- apply(x, 1, probfunct, N = N, n = n, ym = ym, nym = nym)
                                pmax1 <- max(z)
                                M <- x[z == max(z),  ]
                                if(length(M) != length(N)) {
                                  M <- M[1,  ]
                                }
                        }
                        c(pmax1, M)
                }
                width <- max(trunc(width * sum(M)), 2)
                x <- c(0, M)
                while(width > 1) {
                        width <- trunc(width/2)
                        M <- x[-1]
                        x <- search1.function(N, M, n, y, ym, nym, n.matrix, width)
                }
                x
        }
        x <- corner.function(N, n, y, Th, permutations = permutations)
        x <- t(apply(x, 1, search.function, N = N, n = n, y = y, ym = ym, nym = nym, n.matrix = n.matrix, width = width))
        x <- x[x[, 1] == max(x[, 1]),  ]
        x[1,  ]
}
#
ym.function <- function(N, n, y)
# returns a matrix with rows specifying points in the critical region
# used for dermining a p-value given N, n, and y.  The critical region
# is defined as all sample outcomes where phat (see phat.function) is
# less than or equal to the observed phat.
{
        phat <- phat.function(N, n, y)
        phat.vector <- rep(0, length(N))
        for(i in 1:length(N)) {
                phat.vector[i] <- min(trunc((phat * n[i])/N[i]), n[i]) + 1
        }
        ym <- combinations.function(phat.vector) - 1
        ym[apply(ym, 1, phat.function, N = N, n = n) <= phat,  , drop = F]
}
#
ym.prime <- function(N, n, y)
# same as ym.function except the critical region is defined as outcome
# where phat is less than the observed phat.  Used for calculating
# p-values in connection with determining lower confidence bounds.
{
        phat <- phat.function(N, n, y)
        phat.vector <- rep(0, length(N))
        for(i in 1:length(N)) {
                phat.vector[i] <- min(trunc((phat * n[i])/N[i]), n[i]) + 1
        }
        ym <- combinations.function(phat.vector) - 1
        ym[apply(ym, 1, phat.function, N = N, n = n) < phat,  ]
}
#
newym <- function(ym)
# takes the critical region matrix, ym, and converts
# it into a list of vectors where each vector contains
# all possible integer values for y in a particular
# stratum that occur in the critical region.
{
        z <- apply(ym, 2, max)
        ymlist <- vector("list", length(z))
        for(i in seq(along = z)) {
                ymlist[[i]] <- 0:z[i]
        }
        ymlist
}
#
probfunct <- function(N, M, n, ym, nym)
# Calculates the probability that a sample outcome will be
# in the critical region ym given a population
# with strata N, each containing M errors, with samples of size n
# taken for each strata.  See also ym.function and newym.
{
        logprobs <- function(N, M, n, ym, nym)
        {
                lhypergeo <- function(N, M, n, y)
                {
                        length.q <- max(length(N), length(M), length(n), length(y))
                        N <- rep(round(N), length.out = length.q)
                        M <- rep(round(M), length.out = length.q)
                        n <- rep(round(n), length.out = length.q)
                        y <- rep(round(y), length.out = length.q)
                        p <- y
                        lhypergeo2 <- function(N, M, n, y)
                        {
                                lcomb <- function(x, y)
                                {
                                  lgamma(x + 1) - (lgamma(y + 1) + lgamma(x - y + 1))
                                }
                                if(y > M) {
                                  p <-  - Inf
                                }
                                else {
                                  p <- lcomb(M, y) + lcomb(N - M, n - y) - lcomb(N, n)
                                }
                        }
                        for(i in seq(along = y)) {
                                p[i] <- lhypergeo2(N[i], M[i], n[i], y[i])
                        }
                        p
                }
                plist <- vector("list", length(nym))
                for(i in seq(along = nym)) {
                        plist[[i]] <- lhypergeo(N[i], M[i], n[i], nym[[i]])
                }
                plist
        }
        plist <- logprobs(N, M, n, ym, nym)
        ym <- ym + 1
        for(i in 1:nrow(ym)) {
                for(j in 1:ncol(ym)) {
                        ym[i, j] <- plist[[j]][ym[i, j]]
                }
        }
        sum(exp(ym %*% rep(1, ncol(ym))))
}
#
n.matrix.function <- function(x)
# takes a matrix x as input containing positive
# integers and outputs a matrix where x[i,j] = x[i,j]
# if x[i,j] is <= 2, and x[i,j] = 3 otherwise with
# duplicate rows eliminated.
{
        x <- ifelse(x <= 2, x, 3)
        unique.rows(x)
}
#
corner.function <- function(N, n, y, Th, permutations = permute.function(length(N)))
# finds the corner points of the parameter space defined by
# N,n,y, and Th (see definitions of these variables under pmax.function).
{
        startM <- permutations
        for(i in 1:nrow(permutations)) {
                l <- sum(y)
                Tl <- Th
                for(ii in 1:ncol(permutations)) {
                        j <- permutations[i, ii]
                        k <- min(Tl - l + y[j], N[j] - n[j] + y[j])
                        Tl <- Tl - k
                        l <- l - y[j]
                        startM[i, j] <- k
                }
        }
        unique.rows(startM)
}
#
ubnest.function <- function(N, n, y, ym, nym, permutations, n.matrix, cl = 0.95)
# provides ubstrat.function with a starting value for calculating the
# upper confidence bound.
{
        alpha <- 1 - cl
        Th <- ubest.function(N, n, y, cl)
        x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        ratio.v <- x[-1]/Th
        interval <- Th
        while(interval > 1) {
                interval <- interval/2
                p <- probfunct(N, Th * ratio.v, n, ym, nym)
                Th <- ifelse(p > alpha, Th + interval, Th - interval)
        }
        round(Th)
}
#
ubest.function <- function(N, n, y, cl = 0.95)
# provides ubnest.function with a starting value for calculating the
# upper confidence bound.
{
        round(sqrt(sum((ubhyper(N, n, y, cl) - y/n * N)^2)) + phat.function(N, n, y))
}
#
lbnest.function <- function(N, n, y, ym, nym, permutations, n.matrix, cl = 0.95)
# provides lbstrat.function with a starting value for calculating the
# lower confidence bound.
{
        alpha <- 1 - cl
        Th <- lbest.function(N, n, y, cl)
        x <- pmax.short(N, Th, n, y, ym, nym, permutations, n.matrix)
        ratio.v <- x[-1]/Th
        interval <- Th
        while(interval > 1) {
                interval <- interval/2
                p <- 1 - probfunct(N, Th * ratio.v, n, ym, nym)
                Th <- ifelse(p > alpha, Th - interval, Th + interval)
        }
        round(Th)
}
#
lbest.function <- function(N, n, y, cl = 0.95)
# provides lbnest.function with a starting value for calculating the
# lower confidence bound.
{
        round(phat.function(N, n, y) - sqrt(sum((y/n * N - lbhyper(N, n, y, cl))^2)))
}
#
