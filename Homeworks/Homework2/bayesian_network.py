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
cloudy = BayesNet([
    ('Cloudy', '', 0.50),
    ('Rain', 'Cloudy', {T: 0.80, F: 0.20}),
    ('Sprinkler', 'Cloudy', {T: 0.10, F: 0.50}),
    ('WetGrass', 'Sprinkler Rain', {(T, T): 0.99, (T, F): 0.90, (F, T): 0.90, (F, F): 0.00})
    ])

# Compute P(Cloudy)
print("\nP(Cloudy)")
print(enumeration_ask('Cloudy', dict(), cloudy).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Cloudy', dict(), cloudy).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Cloudy', dict(), cloudy).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Cloudy', dict(), cloudy).show_approx())
print(likelihood_weighting('Cloudy', dict(), cloudy).show_approx())

###################################################################################################

# Compute P(Sprinkler | cloudy)
print("\nP(Sprinkler | cloudy)")
print(enumeration_ask('Sprinkler', dict(Cloudy=T), cloudy).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Sprinkler', dict(Cloudy=T), cloudy).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Sprinkler', dict(Cloudy=T), cloudy).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Sprinkler', dict(Cloudy=T), cloudy).show_approx())
print(likelihood_weighting('Sprinkler', dict(Cloudy=T), cloudy).show_approx())

###################################################################################################

# Compute P(Cloudy| the sprinkler is running and it’s not raining)
print("\nP(Cloudy| the sprinkler is running and it’s not raining)")
print(enumeration_ask('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())
print(likelihood_weighting('Cloudy', dict(Sprinkler=T, Rain=F), cloudy).show_approx())

###################################################################################################

# Compute P(WetGrass | it’s cloudy, the sprinkler is running and it’s raining)
print("\nP(WetGrass | it’s cloudy, the sprinkler is running and it’s raining)")
print(enumeration_ask('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())
print(likelihood_weighting('WetGrass', dict(Cloudy=T, Sprinkler=T, Rain=T), cloudy).show_approx())

###################################################################################################

# Compute P(Cloudy | the grass is not wet)
print("\nP(Cloudy | the grass is not wet)")
print(enumeration_ask('Cloudy', dict(WetGrass=F), cloudy).show_approx())
# elimination_ask() is a dynamic programming version of enumeration_ask().
print(elimination_ask('Cloudy', dict(WetGrass=F), cloudy).show_approx())
# gibbs_ask() is an approximation algorithm helps Bayesian Networks scale up.
print(gibbs_ask('Cloudy', dict(WetGrass=F), cloudy).show_approx())
# See the explanation of the algorithms in AIMA Section 14.4.
print(rejection_sampling('Cloudy', dict(WetGrass=F), cloudy).show_approx())
print(likelihood_weighting('Cloudy', dict(WetGrass=F), cloudy).show_approx())
