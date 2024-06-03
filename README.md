# feldman_cousins
Tool for producing confidence intervals from the Feldman-Cousins technique with a Poisson random variable and a known background.

Tip: do a git-clone of this repository into a directory called `feldman_cousins`, and make that directory findable for a python import.  This will reproduce the values in e.g. Feldman-Cousins Table IV, where one specifies an observed number of events and a background expectation.  A confidence level can be provided as well, through the input `alpha` (`alpha=0.1` corresponds to 90% confidence intervals).

To reproduce the `n = 2, b = 3.0` element of Table IV, one would do the following:

```
>>> import feldman_cousins as fc
>>> fc.FC_ints(2, 3.0)
(0.0, 3.035)
```
which is consistent with the interval in the paper.  This function (`FC_ints`) can be used to find confidence intervals for arbitrary _n_ and _b_.

Note: Feldman and Cousins note two "mild pathologies" of their technique which they have remedied.  These are actually not pathologies, just complications.  This code follows their remedy for the first mild pathologies (removing disjoint confidence intervals through overcoverage) but does not address the second mild pathology (the upper limit is not always a decreasing function of _b_ for fixed _n_).
