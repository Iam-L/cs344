'''
This module implements the Bayesian network shown in the text, Figure 14.2.
It's taken from the AIMA Python code.

@author: kvlinden
@version Jan 2, 2013
'''

from probability import BayesNet, enumeration_ask, elimination_ask, gibbs_ask, rejection_sampling, likelihood_weighting

# Utility variables
T, F = True, False

# From AIMA code (probability.py) - Fig. 14.2 - burglary example
burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])

# Compute P(Burglary | John and Mary both call).
print("\nP(Burglary | John and Mary both call)")
print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
print(likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())

# Compute P(Alarm | burglary ∧ ¬earthquake)
print("\nP(Alarm | burglary ∧ ¬earthquake")
print(enumeration_ask('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())
print(likelihood_weighting('Alarm', dict(Burglary=T, Earthquake=F), burglary).show_approx())

# Compute P(John | burglary ∧ ¬earthquake)
print("\nP(John | burglary ∧ ¬earthquake")
print(enumeration_ask('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())
print(likelihood_weighting('JohnCalls', dict(Burglary=T, Earthquake=F), burglary).show_approx())

# Compute P(Burglary | alarm)
print("\nP(Burglary | alarm)")
print(enumeration_ask('Burglary', dict(Alarm=T), burglary).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Burglary', dict(Alarm=T), burglary).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Burglary', dict(Alarm=T), burglary).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Burglary', dict(Alarm=T), burglary).show_approx())
print(likelihood_weighting('Burglary', dict(Alarm=T), burglary).show_approx())

# Compute P(Burglary | john ∧ mary)
print("\nP(Burglary | john ∧ mary")
print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
print(likelihood_weighting('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
