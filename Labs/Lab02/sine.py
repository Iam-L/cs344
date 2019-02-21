"""
This module implements local search on a simple abs function variant.
The function is a linear function  with a single, discontinuous max value
(see the abs function variant in graphs.py).

Note: modified to use a sine function.

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

            # Formulate a problem with a 2D hill function and a single maximum value.
            maximum = 30
            initial = randrange(0, maximum)
            p = AbsVariant(initial, maximum, delta=1.0)
            print('Initial                      x: ' + str(p.initial)
                  + '\t\tvalue: ' + str(p.value(initial))
                  )

            hillclimbstart = time.time()

            # Solve the problem using hill-climbing.
            hill_solution = hill_climbing(p)
            print('Hill-climbing solution       x: ' + str(hill_solution)
                  + '\tvalue: ' + str(p.value(hill_solution))
                  )

            currentHighestClimb = p.value(hill_solution)
            hillclimbend = time.time()

            annealstart = time.time()

            # Solve the problem using simulated annealing.
            annealing_solution = simulated_annealing(
                p,
                exp_schedule(k=20, lam=0.005, limit=1000)
            )
            print('Simulated annealing solution x: ' + str(annealing_solution)
                  + '\tvalue: ' + str(p.value(annealing_solution))
                  )

            annealend = time.time()

            # Print the run-time of each algorithm
            hill_climb_elapsed_time = hillclimbend - hillclimbstart
            anneal_elapsed_time = annealend - annealstart

            print('\nHill-climbing elapsed time is: ' + str(hill_climb_elapsed_time))
            print('Annealing elapsed time is: ' + str(anneal_elapsed_time))