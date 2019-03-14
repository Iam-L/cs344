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

# From Bayesian network in Lab05 - Exercise 5.3
happiness = BayesNet([
    ('Sunny', '', 0.7),
    ('Raise', '', 0.01),
    ('Happy', 'Sunny Raise', {(T, T): 1.0, (T, F): 0.7, (F, T): 0.9, (F, F): 0.1}),
    ])

'''
 Inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you obtain a raise given that it is sunny.
'''

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

'''
 Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you obtain a raise given that you are happy and it is sunny.
'''

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

'''
 Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you obtain a raise given that you are happy.  Depends on sunny as an evidence variable.
'''

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

'''
 Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that you obtain a raise given that you are happy and it is not sunny.
'''

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

'''
The results of part a make sense to me.  There is a higher chance you received a raise if it is not only
sunny, but you are also happy to boot.
'''

'''
The results of part b do make sense.  It is unlikely you will receive a raise just because you are happy.
It is also unlikely you will receive a raise because you are happy and it is not sunny.

As for why the probability of a raise given that you are happy and it is not sunny is higher than the probability
of a raise given that you are happy, I wouldn't naturally infer them using "common sense".
'''

'''
The results of the exact inference algorithms versus the approximate inference algorithms do differ by some margin.
Our hand calculations almost always gives the same exact numbers as the exact inference algorithms to a certain
degree of precision.  The results doesn't match exactly probably because approximate inference algorithms trade 
accuracy for decreased computation time while exact inference algorithms go for accuracy at the expense of 
computational time.
'''

