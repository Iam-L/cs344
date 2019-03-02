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

# From AIMA code (probability.py) - Fig. 14.2 - burglary example
burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])

'''
Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that a burglary occurs given that john and mary both call, which depends on alarm and earthquake as
evidence variables.
'''

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

'''
Causal inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that the alarm goes off given that a burglary occurs and there is no earthquake.
'''

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

'''
Causal inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that john calls given that a burglary occurs and there is no earthquake.  Depends on alarm
as an evidence variable.
'''

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

'''
Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that a burglary occurs given that the alarm goes off.  Depends on earthquake as an evidence
variable.
'''

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

'''
Diagnostic inference.  Refer to screen capture and/or turned in paper copy for mathematical explanation.

The probability that a burglary occurs given that john and mary both call, which depends on alarm and earthquake as
evidence variables.
'''

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

'''
the probability that john calls given a burglary and no earthquake is lower than the probability that the alarm
goes off given a burglary and no earthquake.  I suppose that makes sense as john calling is one level deeper than
the alarm going off, which means there are more combinations of factors affecting john than the alarm.

The probability of a burglary given that the alarm goes off does make sense as the probability of an earthquake is
higher, so if the alarm goes off there is a higher chance it is because there is an earthquake.

I'm surprised that the probability that a burglary has occurred given that john and mary calls isn't higher.
Then again, the probability that an earthquake occurs is higher than that of a burglary by 0.001, so I suppose
it makes sense that if john and mary calls it is more likely to be about an earthquake.

'''

'''
The results of the exact inference algorithms versus the approximate inference algorithms do differ by some margin.
Our hand calculations almost always gives the same exact numbers as the exact inference algorithms to a certain
degree of precision.  The results doesn't match exactly probably because approximate inference algorithms trade 
accuracy for decreased computation time while exact inference algorithms go for accuracy at the expense of 
computational time.
'''
