"""
Name: Joseph Jinn
Date: 2-22-19
Course: CS-344 Artificial Intelligence
Instructor: Professor Keith VanderLinden
Assignment: Lab04 - Probability

Modified from joint.py provided by Professor VanderLinden and AIMA.

Note:

Discrete Mathematics ftw.

"""

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
print("True = Heads; False = Tails")
PC = enumerate_joint_ask('Coin2', {'Coin1': T}, P)
print(PC.show_approx())
print("\n")

"""
Exercise 4.1

#############################################################################################
Compute the value of P(Cavity|catch) :

First, compute it by hand.


P(Cavity | Catch) = P(Cavity and Catch) / P(Catch) = 

= 0.5294117647 (using Maple software to do the actual arithmetic)


##############################################
Verify your answer (and the AIMA implementation) by adding code to compute the specified value.


Probability of Cavity | Catch
False: 0.471, True: 0.529


#############################################################################################
Does the answer confirm what you believe to be true about the probabilities of flipping coins?


Probability of Coin2 | Coin1 = Heads
True = Heads; False = Tails
False: 0.5, True: 0.5


Yes, they are independent events so the outcome of one does not affect the outcome of the other.
Hence, it is still a 50 percent chance of a heads or tails.
(provided I programmed the joint probability distribution function correctly)


#############################################################################################
Can you see now why the full joint is generally not used in probabilistic systems?

When a probabilistic query has more than one piece of evidence
the approach based on full joint probability will not scale up
P(Cavity | toothache  catch)

• Neither will applying Bayes’ rule scale up in general
Ĵ P(toothache  catch | Cavity) P(Cavity)

• We would need variables to be independent, but variable
Toothache and Catch obviously are not: if the probe catches in
the tooth, it probably has a cavity and that probably causes a
toothache

• Each is directly caused by the cavity, but neither has a direct
effect on the other

• catch and toothache are conditionally independent given Cavity

URL: https://www.cs.tut.fi/~elomaa/teach/AI-2012-7.pdf

I guess that explains it?


#############################################################################################
"""
