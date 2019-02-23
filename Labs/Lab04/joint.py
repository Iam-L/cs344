'''
This module implements a simple classroom example of probabilistic inference
over the full joint distribution specified by AIMA, Figure 13.3.
It is based on the code from AIMA probability.py.

@author: kvlinden
@version Jan 1, 2013
'''

#############################################################################################
#############################################################################################

from probability import JointProbDist, enumerate_joint_ask

# The Joint Probability Distribution Fig. 13.3 (from AIMA Python)
P = JointProbDist(['Toothache', 'Cavity', 'Catch'])
T, F = True, False
P[T, T, T] = 0.108
P[T, T, F] = 0.012
P[F, T, T] = 0.072
P[F, T, F] = 0.008
P[T, F, T] = 0.016
P[T, F, F] = 0.064
P[F, F, T] = 0.144
P[F, F, F] = 0.576

# Compute P(Cavity|Toothache=T)  (see the text, page 493).
print("Probability of Cavity | Toothache")
PC = enumerate_joint_ask('Cavity', {'Toothache': T}, P)
print(PC.show_approx())
print("\n")

print("Probability of Cavity | Catch")
PC = enumerate_joint_ask('Cavity', {'Catch': T}, P)
print(PC.show_approx())
print("\n")

#############################################################################################
#############################################################################################

P = JointProbDist(['Coin1', 'Coin2'])
T, F = True, False
P[T, T] = 0.5 ** 2
P[F, T] = 0.5 ** 2
P[T, F] = 0.5 ** 2
P[F, F] = 0.5 ** 2

print("Probability of Coin2 | Coin1 = Heads")
PC = enumerate_joint_ask('Coin2', {'Coin1': T}, P)
print(PC.show_approx())
print("\n")

"""
Exercise 4.1

Compute the value of P(Cavity|catch) by hand:

Does the answer confirm what you believe to be true about the probabilities of flipping coins?

Can you see now why the full joint is generally not used in probabilistic systems?
"""

"""

"""