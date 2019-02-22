"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Traveling Salesman Local Search Problem

Note:

Don't over-complicate things.

My Python code is inefficient AF due to lack of Python familiarity.
I did try to clean it up a bit regardless.

Also, as I am using a random number generator for the distances for each pair of cities, it is
impossible to truly determine whether the formulation is 100% correct as the output necessarily
differs for every execution of the program.

Both hill-climbing and simulated annealing consistently finds solutions, optimal or not, for
the traveling salesman problem.

"""

############################################################################################
############################################################################################

import itertools
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
    """ Initialize class variables. """

    def __init__(self, distances, cities):
        self.distances = distances
        # Note: must have self.initial as defined in search.py
        self.initial = cities

    ################################################################################################

    """ Swap the position of two cities at random. """

    def actions(self, state):
        # Turn debug statements on or off.
        debug_actions = False

        # Store possible actions.
        actions = []

        # Randomly pick cities to swap based on their indices.
        for i in range(0, 10):
            # Selects pairs of cities of specified size at random from set of cities.
            sample_size = 2
            my_sample = random.sample(range(len(state)), sample_size)

            if debug_actions:
                # Debug statement.
                print("My random sample: " + str(my_sample))

            actions.append(my_sample)

        if debug_actions:
            # Debug statement.
            print("\nAll possible actions: " + str(actions))

        return actions

    ################################################################################################

    """Makes the given move on a copy of the given state."""

    def result(self, state, action):

        # Turn debug statements on or off.
        debug_result = False

        # Copy the state to a new state.
        new_state = state

        if debug_result:
            print("\nChosen cities (as a tuple): " + str(action))
            print("Chosen city A: " + str(action[0]))
            print("Chosen city B: " + str(action[1]))

        # Rename just for my own clarity of purpose.
        city_a = int(action[0])
        city_b = int(action[1])

        # Convert to list as sets don't support subscripting.
        new_state_as_list = list(new_state)
        state_as_list = list(state)

        # Does the actual swapping.
        new_state_as_list[city_a] = state_as_list[city_b]
        new_state_as_list[city_b] = state_as_list[city_a]

        if debug_result:
            print("\nOld state was: " + str(state_as_list))
            print("\nNew state is: " + str(new_state_as_list))

        return set(new_state_as_list)

    ################################################################################################

    """ Compute the total distance of the circuit. """

    def value(self, state):

        # Turn debug statements on or off.
        debug_value = False

        # Convert to list as sets don't support subscripting.
        state_as_list = list(state)

        if debug_value:
            print("\nContents of state: " + str(state_as_list) + "\n")

        # Determine the pairs of cities in circuit to determine distances for each.
        city_pairs = []

        for i in range(len(state) - 1):

            first = state_as_list[i]
            second = state_as_list[i + 1]

            city_pairs.append((str(first), str(second)))

            if debug_value:
                print("(City A, City B): " + first + ", " + second)

        # So that we take into account the distance from last city back to the starting city.
        back_to_origin = (str(state_as_list[len(state_as_list) - 1]), str(state_as_list[0]))
        city_pairs.append(back_to_origin)

        # Debug - check that we have accounted for all pairs of cities whose distances we need.
        if debug_value:
            print("\nContents of city_pairs object: " + str(city_pairs))

        # Debug - check that we have the distance values we need for each pair of cities.
        if debug_value:
            print("\nContents of city distances object: " + str(self.distances))

        # Store the distance of the circuit.
        total_distance = 0

        # Calculate the total distance by summing distances for each pair of cities.
        for pairs in city_pairs:
            # print("Checking pair: " + str(pairs))

            for _key, _value in self.distances.items():
                # print("Checking key: " + str(key))

                if str(pairs) == str(_key):
                    total_distance += _value

        if debug_value:
            print("\nThe total distance for the circuit is: " + str(total_distance))

        return total_distance


################################################################################################
################################################################################################

# Utility function to generate a random number to be used as distance between cities.
def get_rng():
    return random.randint(1, 1000)


""" Attempt to solve a sample Traveling Salesman Problem and print solution to console. """

if __name__ == '__main__':

    # Turn debug statements on or off.
    debug = False

    # Create simple list of cities to travel to.
    cities = {'City1', 'City2', 'City3', 'City4', 'City5', 'City6', 'City7', 'City8'}

    if debug:
        # Debug statement.
        print('\nMy cities: ' + str(cities) + "\n")

    # Create dictionary of distances between all possible pairs of cities A --> B only).
    cityPairs = itertools.combinations(cities, 2)
    cityDistancesUnique = {}
    for element in cityPairs:
        cityDistancesUnique[element] = get_rng()

        if debug:
            print("City Tuple: " + str(element))

        reversedTuple = (element[1], element[0])

        if debug:
            print("Reversed City Tuple: " + str(reversedTuple))

        cityDistancesUnique[reversedTuple] = cityDistancesUnique[element]

    if debug:
        print("\nContents of cityDistancesUnique: " + str(cityDistancesUnique))

    if debug:
        # Debug statement.
        print("\nDistances between pairs of cities: A --> B and B --> A")
        for key, value in cityDistancesUnique.items():
            print("Key is: " + str(key))
            print("Value is: " + str(value))

    ################################################################################################

    # Initialize the Traveling Salesman Problem.
    travel = Salesman(cityDistancesUnique, cities)

    # Solve the problem using hill climbing.
    hillStartTime = time.time()
    hill_solution = hill_climbing(travel)
    hillEndTime = time.time()
    print('Hill-climbing:')
    print('\tSolution: ' + str(hill_solution))
    print('\tValue:    ' + str(travel.value(hill_solution)))
    print('\tTime to find solution using hill-climbing' + str(hillEndTime - hillStartTime))

    # Solve the problem using simulated annealing.
    annealStartTime = time.time()
    annealing_solution = simulated_annealing(travel,
                                             exp_schedule(k=20, lam=0.005, limit=10000))
    annealEndTime = time.time()
    print('Simulated annealing:')
    print('\tSolution: ' + str(annealing_solution))
    print('\tValue:    ' + str(travel.value(annealing_solution)))
    print('\tTime to find solution using simulated annealing' + str(annealEndTime - annealStartTime))

    ################################################################################################
    ################################################################################################
