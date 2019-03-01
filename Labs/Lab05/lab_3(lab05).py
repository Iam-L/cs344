'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden
@version Jan 2, 2013
'''

from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, rejection_sampling, likelihood_weighting

# Utility variables
T, F = True, False

# From Bayesian network in Lab05 - Exercise 5.3
happiness = BayesNet([
    ('Sunny', '', 0.7),
    ('Raise', '', 0.01),
    ('Happy', 'Sunny Raise', {(T, T): 1.0, (T, F): 0.7, (F, T): 0.9, (F, F): 0.1}),
    ])

# Compute P(Raise | sunny)
print("\nP(Raise | sunny)")
print(enumeration_ask('Raise', dict(Sunny=T), happiness).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Raise', dict(Sunny=T), happiness).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Raise', dict(Sunny=T), happiness).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Raise', dict(Sunny=T), happiness).show_approx())
print(likelihood_weighting('Raise', dict(Sunny=T), happiness).show_approx())

# Compute P(Raise | happy ∧ sunny)
print("\nP(Raise | happy ∧ sunny)")
print(enumeration_ask('Raise', dict(Happy=T, Sunny=T), happiness).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Raise', dict(Happy=T, Sunny=T), happiness).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Raise', dict(Happy=T, Sunny=T), happiness).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Raise', dict(Happy=T, Sunny=T), happiness).show_approx())
print(likelihood_weighting('Raise', dict(Happy=T, Sunny=T), happiness).show_approx())

# Compute P(Raise | happy)
print("\nP(Raise | happy)")
print(enumeration_ask('Raise', dict(Happy=T), happiness).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Raise', dict(Happy=T), happiness).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Raise', dict(Happy=T), happiness).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Raise', dict(Happy=T), happiness).show_approx())
print(likelihood_weighting('Raise', dict(Happy=T), happiness).show_approx())

# Compute P(Raise | happy ∧ ¬sunny)
print("\nP(Raise | happy ∧ ¬sunny)")
print(enumeration_ask('Raise', dict(Happy=T, Sunny=F), happiness).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Raise', dict(Happy=T, Sunny=F), happiness).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Raise', dict(Happy=T, Sunny=F), happiness).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Raise', dict(Happy=T, Sunny=F), happiness).show_approx())
print(likelihood_weighting('Raise', dict(Happy=T, Sunny=F), happiness).show_approx())
