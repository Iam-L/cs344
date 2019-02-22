"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Course Scheduling Constraint Satisfaction Problem

Notes:

None at the moment.

"""

############################################################################################
############################################################################################

# Time how long each algorithm takes.
import time

# Import everything because I'm lazy and inefficient. ;D
import csp
import search

import time

############################################################################################
############################################################################################

# Turn debugging on or off.
debug = False


# Define a course scheduling formulation.
def courses():
    # Defines the variables and values.
    Courses = 'cs108 cs112 cs212 cs214 cs232 cs262 cs344'.split()
    Faculty = 'adams vanderlinden plantinga wieringa norman'.split()
    TimeSlots = 'mwf8:00-8:50 mwf9:00-9:50 mwf10:30-11:20 tth11:30-12:20 tth12:30-1:20'.split()
    Classrooms = 'nh253 sb382'.split()

    if debug:
        # Debug statements.
        print('Courses list:' + str(Courses))
        print('Faculty list:' + str(Faculty))
        print('TimeSlots list:' + str(TimeSlots))
        print('Classrooms list:' + str(Classrooms))

    variables = Courses
    values = Faculty + TimeSlots + Classrooms

    if debug:
        # Debug statements.
        print('My variables: ' + str(variables))
        print('My values: ' + str(values))

    # Defines the domain.
    domain = {}
    for var in variables:
        domain[var] = values

    if debug:
        # Debug statements.
        for key, value in domain.items():
            print("My domain key: " + key)
            print("My domain values: " + str(value))

    # Define neighbors of each variable.
    neighbors = csp.parse_neighbors("""cs108: cs112 cs212 cs214 cs232 cs262 cs344;
                cs112: cs108 cs212 cs214 cs232 cs262 cs344; 
                cs212: cs108 cs112 cs214 cs232 cs262 cs344; 
                cs214: cs108 cs112 cs212 cs232 cs262 cs344;
                cs232: cs108 cs112 cs212 cs214 cs262 cs344; 
                cs262: cs108 cs112 cs212 cs214 cs232 cs344; 
                cs344: cs108 cs112 cs212 cs214 cs232 cs262""")

    if debug:
        # Debug statements.
        for key, value in neighbors.items():
            print("Neighbors Key:" + key)
            print("Neighbors Values: " + str(value))

    """
        The constraints are that:
            each course should be offered exactly once by the assigned faculty member.
            a faculty member can only teach one thing at a time.
            a room can only have one class at each time.
    """

    # Define the constraints on the variables.
    # FIXME - I'm obviously not understanding how this is supposed to work.
    def scheduling_constraint(A, a, B, b):

        # Fail if in the same room at the same time.
        for myRoom in Classrooms:
            if A == myRoom and B == myRoom:
                for myTime in TimeSlots:
                    if A == myTime and B == myTime:
                        return False

        # Fail if the same faculty at the same time.
        for myTeacher in Faculty:
            if A == myTeacher and B == myTeacher:
                for myTime in TimeSlots:
                    if A == myTime and B == myTime:
                        return False

        return True
        # raise Exception('error')

    return csp.CSP(variables, domain, neighbors, scheduling_constraint)


############################################################################################
############################################################################################

# Assign problem definition to variable.
problem = courses()

""" Select a AIMA search algorithm to use to solve the problem and time its runtime. """
startTime = time.time()

# result = search.depth_first_graph_search(problem)
# result = csp.AC3(problem)
# result = csp.backtracking_search(problem)
result = csp.min_conflicts(problem, max_steps=1000)

endTime = time.time()

""" A CSP solution printer copied from csp.py. """


def print_solution(my_results):
    for h in range(1, 6):
        print('House', h)
        for (var, val) in my_results.items():
            if val == h:
                print('\t', var)


""" Print the solution. """

if problem.goal_test(problem.infer_assignment()):
    print("Solution:\n")
    print_solution(result)
else:
    print("failed...")
    print(problem.curr_domains)
    problem.display(problem.infer_assignment())

    print("Time taken to execute algorithm: " + str(endTime - startTime))

############################################################################################
############################################################################################
