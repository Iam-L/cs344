"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Traveling Salesman Local Search Problem

Note to self:

FIXME - errors galore!

"""

############################################################################################
############################################################################################

# Time how long each algorithm takes.
import time
# Random number generator.
import random

# Import the search algorithms we will use, cooling function, etc.
from search import Problem, hill_climbing, simulated_annealing, exp_schedule

############################################################################################
############################################################################################

"""
Defines the Traveling Salesman problem by modifying queens.py from u02local directory.
"""


class Salesman(Problem):
    """ Initalize class variables. """

    def __init__(self, distances, city):
        self.distances = distances
        # Note: must have self.initial as defined in search.py
        self.initial = city

    """ Swap the position of two cities at random. """

    def actions(self, state):
        actions = []
        c = state

        # Does the actual city swapping.
        key1, key2 = random.sample(list(c), 2)
        c[key1], c[key2] = c[key2], c[key1]

        actions.append(c)
        return actions

    """Makes the given move on a copy of the given state."""

    def result(self, state, action):
        new_state = state[:]
        new_state = action
        return new_state

    """ Compute the total distance of the circuit. """

    def value(self, state):

        # Pseudo-code.
        # Take the state (list of cities in its current order)
        # Determine the pairs of cities in the circuit from start to end.
        # Refer to defined distances between pairs of cities in dictionary.
        # Calculate the distance traveled for this circuit.
        # Return the result.

        definedDistances = self.distances
        totalDistance = 0




        return totalDistance


################################################################################################
################################################################################################

# Utility function to generate a random number to be used as distance between cities.
def get_rng():
    return random.randint(1, 100)


""" Attempt to solve a sample Traveling Salesman Problem and print solution to console. """

if __name__ == '__main__':

    # Create simple list of cities to travel to.
    cities = {'City1', 'City2', 'City3', 'City4'}

    # Debug statement.
    print('My cities:    ' + str(cities))

    # Create dictionary of distances between all possible pairs of cities.
    cityDistances = {}
    tuples = {}
    for c1 in cities:
        for c2 in cities:
            if c1 != c2:
                cityDistances[(c1, c2)] = get_rng()

    # Debug statement.
    for key, value in cityDistances.items():
        print("Key is: " + str(key))
        print("Value is: " + str(value))

    # Initialize the Traveling Salesman Problem.
    travel = Salesman(cityDistances, cities)

    # Solve the problem using hill climbing.
    hillStartTime = time.time()
    hill_solution = hill_climbing(travel)
    hillEndTime = time.time()
    print('Hill-climbing:')
    print('\tSolution: ' + str(hill_solution))
    print('\tValue:    ' + str(travel.value(hill_solution)))
    print('\tGoal?     ' + str(travel.goal_test(hill_solution)))
    print('\tTime to find solution using hill-climbing' + str(hillEndTime - hillStartTime))

    # Solve the problem using simulated annealing.
    annealStartTime = time.time()
    annealing_solution = simulated_annealing(travel,
                                             exp_schedule(k=20, lam=0.005, limit=10000))
    annealEndTime = time.time()
    print('Simulated annealing:')
    print('\tSolution: ' + str(annealing_solution))
    print('\tValue:    ' + str(travel.value(annealing_solution)))
    print('\tGoal?     ' + str(travel.goal_test(annealing_solution)))
    print('\tTime to find solution using simulated annealing' + str(annealEndTime - annealStartTime))

    ################################################################################################
    ################################################################################################
