'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden
@version Jan 2, 2013

Note: Refer to screen captures and/or turned in paper copy for hand calculation work.
'''

from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, rejection_sampling, likelihood_weighting

# Utility variables
T, F = True, False

# From Bayesian network in Lab05 - Exercise 5.2
cancer = BayesNet([
    ('Cancer', '', 0.01),
    ('Test1', 'Cancer', {T: 0.90, F: 0.2}),
    ('Test2', 'Cancer', {T: 0.90, F: 0.2})
    ])

'''
Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you have cancer given that you obtained a positive result on both tests.
'''

# Compute P(Cancer | positive results on both tests)
print("\nP(Cancer | positive results on both tests)")
print(enumeration_ask('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())
print(likelihood_weighting('Cancer', dict(Test1=T, Test2=T), cancer).show_approx())

'''
Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you have cancer given that you obtained a positive result on one test and a negative result
on the other test.
'''

# Compute P(Cancer | a positive result on test 1, but a negative result on test 2)
print("\nP(Cancer | a positive result on test 1, but a negative result on test 2)")
print(enumeration_ask('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())
print(likelihood_weighting('Cancer', dict(Test1=T, Test2=F), cancer).show_approx())

'''
The results make sense.  If you have a positive screening on both tests it stands to reason that you are much more
likely to actually have cancer than if you have a positive on one test and a negative on the other test.

One failed test drastically lowers your probability of actually have cancer from 0.17 --> 0.00565, which is
17% chance to 0.565% chance.
'''

'''
The results of the exact inference algorithms versus the approximate inference algorithms do differ by some margin.
Our hand calculations almost always gives the same exact numbers as the exact inference algorithms to a certain
degree of precision.  The results doesn't match exactly probably because approximate inference algorithms trade 
accuracy for decreased computation time while exact inference algorithms go for accuracy at the expense of 
computational time.
'''

