"""
This module implements local search on a simple abs function variant.
The function is a linear function  with a single, discontinuous max value
(see the abs function variant in graphs.py).

Note: modified to use a sine function and random restarts.

@author: kvlinden
@version 6feb2013
"""
from tools.aima.search import Problem, hill_climbing, simulated_annealing, \
    exp_schedule, genetic_search
from random import randrange
import math
import time


class AbsVariant(Problem):
    """
    State: x value for the abs function variant f(x)
    Move: a new x value delta steps from the current x (in both directions)
    """

    def __init__(self, initial, maximum=30.0, delta=1.0):
        self.initial = initial
        self.maximum = maximum
        self.delta = delta

    def actions(self, state):
        return [state + self.delta, state - self.delta]

    def result(self, stateIgnored, x):
        return x

    # modified to use the sine function variant as a objective function.
    def value(self, x):
        # return self.maximum / 2 - math.fabs(self.maximum / 2 - x)
        return abs(x * math.sin(x))


if __name__ == '__main__':

    # Variables for random restart implementation.
    highestClimbValue = 0
    highestClimbSolution = 0
    highestAnnealValue = 0
    highestAnnealSolution = 0

    #####################################################################################

    # Formulate a problem with a 2D hill function and a single maximum value.
    maximum = 30

    #####################################################################################

    hillclimbstart = time.time()

    # Implement random restarts (This could be the entirely wrong way to go about it...)
    hillcounter = 0

    while hillcounter <= 10:
        # Pick a new random initial x-value each time.
        initial = randrange(0, maximum)
        p = AbsVariant(initial, maximum, delta=1.0)
        print('Initial                      x: ' + str(p.initial)
              + '\t\tvalue: ' + str(p.value(initial))
              )

        # Solve the problem using hill-climbing.
        hill_solution = hill_climbing(p)

        # track current results.
        currentClimbValue = hill_solution
        currentClimbSolution = p.value(hill_solution)

        # determine the best solution.
        if currentClimbSolution > highestClimbSolution:
            highestClimbSolution = currentClimbSolution
            highestClimbValue = currentClimbValue

        hillcounter += 1

    hillclimbend = time.time()

    #####################################################################################

    annealstart = time.time()

    # Implement random restarts (This could be the entirely wrong way to go about it...)
    annealcounter = 0

    while annealcounter <= 10:
        # Pick a new random initial x-value each time.
        initial = randrange(0, maximum)
        p = AbsVariant(initial, maximum, delta=1.0)
        print('Initial                      x: ' + str(p.initial)
              + '\t\tvalue: ' + str(p.value(initial))
              )

        # Solve the problem using simulated annealing.
        annealing_solution = simulated_annealing(
            p,
            exp_schedule(k=20, lam=0.005, limit=1000)
        )

        # track current results.
        currentAnnealValue = annealing_solution
        currentAnnealSolution = p.value(annealing_solution)

        # determine the best solution.
        if currentAnnealSolution > highestAnnealSolution:
            highestAnnealSolution = currentAnnealSolution
            highestAnnealValue = currentAnnealValue

        annealcounter += 1

    annealend = time.time()

    #####################################################################################

    # Print the run-time of each algorithm
    hill_climb_elapsed_time = hillclimbend - hillclimbstart
    anneal_elapsed_time = annealend - annealstart

    print('\nHill-climbing elapsed time is: ' + str(hill_climb_elapsed_time))
    print('Annealing elapsed time is: ' + str(anneal_elapsed_time))

    #####################################################################################

    # Print the results of each search algorithm.
    print('\n\nHill-climbing solution       x: ' + str(highestClimbValue)
          + '\tvalue: ' + str(highestClimbSolution)
          )

    print('Simulated annealing solution x: ' + str(highestAnnealValue)
          + '\tvalue: ' + str(highestAnnealSolution)
          )
