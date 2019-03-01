'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden
@version Jan 2, 2013
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
