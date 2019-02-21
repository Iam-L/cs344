"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Traveling Salesman Local Search Problem

Note to self:

FIXME - don't comprehend where to even start...

"""

############################################################################################
############################################################################################

# Time how long each algorithm takes.
import time

import random

# Import everything because I'm lazy and inefficient. ;D
import csp
import search

############################################################################################
############################################################################################

from search import Problem, hill_climbing, simulated_annealing, exp_schedule
import itertools
import math


# Define the traveling salesman problem.
from utils import distance


class Salesman(Problem):

    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        Problem.__init__(self, initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or search.infinity)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = search.infinity
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return search.infinity


# Executes the problem and finds a solution, if any.
if __name__ == '__main__':

    # Generate a random number to be used as distance between cities.
    randomInt = random.randint(0, 100)

    # Create simple map of cities to travel to.
    cities_map = search.UndirectedGraph(dict(
        CityA=dict(CityB=randomInt, CityC=randomInt, CityD=randomInt),
        CityB=dict(CityC=randomInt, CityD=randomInt),
        CityC=dict(CityD=randomInt)))

    # Debug. - need to add/find methods to print out nodes and connections.
    print("Map: " + str(cities_map))

    # Initialize the Traveling Salesman Problem.
    travel = Salesman(cities_map.nodes[0], cities_map.nodes[0], cities_map)

    # Solve the problem using hill climbing.
    hill_solution = hill_climbing(travel)
    print('Hill-climbing:')
    print('\tSolution: ' + str(hill_solution))
    print('\tValue:    ' + str(travel.value(hill_solution)))
    print('\tGoal?     ' + str(travel.goal_test(hill_solution)))

    # Solve the problem using simulated annealing.
    annealing_solution = simulated_annealing(
        p,
        exp_schedule(k=20, lam=0.005, limit=10000)
    )
    print('Simulated annealing:')
    print('\tSolution: ' + str(annealing_solution))
    print('\tValue:    ' + str(travel.value(annealing_solution)))
    print('\tGoal?     ' + str(travel.goal_test(annealing_solution)))
