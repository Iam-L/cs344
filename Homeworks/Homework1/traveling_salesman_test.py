"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Traveling Salesman Local Search Problem

"Unit" Testing Class.

Note:

I have optimized the actual code in traveling_salesman.py.  This was mostly development/testing code.

Different from actual code in traveling_salesman.py as modifications were necessary to make
it run with the actual algorithm (naming conventions, misconceptions, corrections, etc.)

Discovered my misconception about the result function receiving only one action versus
the entire list of possible actions that was determined in the actions function.

"""

############################################################################################
############################################################################################

import itertools
import random

from Homeworks.Homework1.traveling_salesman import get_rng


############################################################################################
############################################################################################


# Test that actions are returning a suitable set of possible city swaps as a list of tuples.
def test_actions(state):
    # Turn debug statements on or off.
    debug_actions = True

    # Store possible actions.
    actions = []

    # Randomly pick cities to swap based on their indices.
    for i in range(0, 10):
        sample_size = 2
        my_sample = random.sample(range(len(state)), sample_size)

        if debug_actions:
            print("\nMy random sample: " + str(my_sample))

        actions.append(my_sample)

    if debug_actions:
        print("\nAll possible actions: " + str(actions))

    return actions


# Test that result is returning the proper new state after performing a possible action.
def test_result(state, possible_actions):
    # Turn debug statements on or off.
    debug_result = True

    # Copy the state to a new state. (make it it copies, and not just redirects pointers)
    new_state = list(state)

    # Randomly choose a swap to perform on the list of cities.
    sample_size = 1
    my_sample = random.sample(range(len(possible_actions)), sample_size)

    if debug_result:
        print("\nAll possible cities that can be chosen to be swapped.")
        print(possible_actions)

    if debug_result:
        print("\nCities chosen to be swapped (index value with retarded parentheses): " + str(my_sample))

    # Hack-ey AF way of removing annoying parentheses from indices.
    for elements in my_sample:
        my_sample = elements

    if debug_result:
        print("\nCities chosen to be swapped (index value without retarded parentheses): " + str(my_sample))

    # Store the city pair tuple that was chosen in a new list.
    my_chosen_cities = possible_actions[my_sample]

    if debug_result:
        print("\nChosen cities (as a tuple): " + str(my_chosen_cities))
        print("Chosen city A: " + str(my_chosen_cities[0]))
        print("Chosen city B: " + str(my_chosen_cities[1]))

    # Ensure that subscripts are just an integer value (due to prior issue with retarded parentheses)
    cityA = int(my_chosen_cities[0])
    cityB = int(my_chosen_cities[1])

    if debug_result:
        print("\nCity A as stored solely as an integer: " + str(cityA))
        print("City B stored solely as an integer: " + str(cityB))

    # Does the actual city swapping. (had to convert to list as sets don't support subscripting)
    new_state_as_list = list(new_state)
    state_as_list = list(state)

    new_state_as_list[cityA] = state_as_list[cityB]
    new_state_as_list[cityB] = state_as_list[cityA]

    if debug_result:
        print("\nOld state was: " + str(state_as_list))
        print("\nNew state is: " + str(new_state_as_list))

    return set(new_state_as_list)


# Test that we are calculating the proper total distance for the circuit.
# TODO - finish.
def test_value(distances, state):
    # Pseudo-code.
    # Take the state (list of cities in its current order)
    # Determine the pairs of cities in the circuit from start to end.
    # Refer to defined distances between pairs of cities in dictionary.
    # Calculate the distance traveled for this circuit.
    # Return the result.

    # Turn debug statements on or off.
    debug_value = True

    # Store the distance of the circuit.
    total_distance = 0

    # Determine the pairs of cities in circuit to determine distances for each.
    city_pairs = []
    state_as_list = list(state)

    if debug_value:
        print("\nContents of state: " + str(state_as_list))

    for i in range(len(state) - 1):
        first = state_as_list[i]
        second = state_as_list[i + 1]
        city_pairs.append((str(first), str(second)))

        if debug_value:
            print("(City A, City B): " + first + ", " + second)

    # So that we take into account the distance from last city back to the starting city.
    back_to_origin = (str(state_as_list[len(state_as_list) - 1]), str(state_as_list[0]))
    city_pairs.append(back_to_origin)

    if debug_value:
        print("\nContents of city_pairs object: " + str(city_pairs))

    # Debug - check that we have the distances we need for each pair of cities.
    if debug_value:
        print("\nContents of city distances object: " + str(distances))

    # Calculate the total distance by summing distances for each pair of cities.
    for pairs in city_pairs:
        # print("Checking pair: " + str(pairs))
        for key, value in distances.items():
            # print("Checking key: " + str(key))
            if str(pairs) == str(key):
                total_distance += value

    if debug_value:
        print("\nThe total distance for the circuit is: " + str(total_distance))

    return total_distance


""" Perform the various tests when file is executed. """
if __name__ == '__main__':

    # Turn debug statements on or off.
    debug = True

    # Create simple list of cities to travel to.
    cities = {'City1', 'City2', 'City3', 'City4'}

    if debug:
        # Debug statement.
        print('My cities:    ' + str(cities))

    # Create dictionary of distances between all possible pairs of cities A --> B only).
    # TODO - using this method to create data structure to store pairs of cities and their distances.
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
        print("Contents of cityDistancesUnique: " + str(cityDistancesUnique))

    if debug:
        # Debug statement.
        print("Distances between pairs of cities: A --> B ")
        for key, value in cityDistancesUnique.items():
            print("Key is: " + str(key))
            print("Value is: " + str(value))

    ############################################################################################

    # Test whether actions works properly.
    myActions = test_actions(cities)

    # Test whether results works properly.
    myResults = test_result(cities, myActions)

    # Test whether value works properly.
    myValue = test_value(cityDistancesUnique, myResults)

############################################################################################
############################################################################################

