"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Course Scheduling Constraint Satisfaction Problem

Notes:

Only succeeds when using backtracking search (consistently gives same solution)

All other algorithms consistently fail miserably.

"""

############################################################################################
############################################################################################

# Time how long each algorithm takes.
import time

# Import everything because I'm lazy and inefficient. ;D
import csp
import search

############################################################################################
############################################################################################

# Turn debugging on or off.
debug = False


# Define a course scheduling formulation.
def courses():
    # Defines the variables and values.
    courses = 'cs108 cs112 cs212 cs214 cs232 cs262 cs344'.split()
    faculty = 'adams vanderlinden plantinga wieringa norman'.split()
    time_slots = 'mwf8:00-8:50 mwf9:00-9:50 mwf10:30-11:20 tth11:30-12:20 tth12:30-1:20'.split()
    classrooms = 'nh253 sb382'.split()

    if debug:
        # Debug statements.
        print('\ncourses list:' + str(courses))
        print('faculty list:' + str(faculty))
        print('time_slots list:' + str(time_slots))
        print('classrooms list:' + str(classrooms))

    variables = courses
    values = faculty + time_slots + classrooms

    if debug:
        # Debug statements.
        print('\nMy variables: ' + str(variables))
        print('My values: ' + str(values))

    # Combine values into triplets for use as part of domain.
    value_triplets = []

    for faculty in faculty:
        for timeslots in time_slots:
            for classroom in classrooms:
                triplet = faculty + ' ' + timeslots + ' ' + classroom
                value_triplets.append(triplet)

    if debug:
        # Debug statement.
        print("\nContents of value_triplets: " + str(value_triplets) + "\n")

    # Defines the domain.
    domain = {}
    for var in variables:
        domain[var] = value_triplets

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
        print("\n\n")
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
    # FIXME - WTB more documentation for AIMA code.
    def scheduling_constraint(A, a, B, b):

        if debug:
            # Debug statement.
            print("\nvalue of A: " + A)
            print("value of B: " + B)
            print("value of a: " + a)
            print("value of b: " + b)

        # Split "a" and "b" from triplets into singlets to test for same'ness.
        a_split = str(a).split()
        b_split = str(b).split()

        if debug:
            # Debug statement.
            print("\na split contents: " + str(a_split))
            print("b split contents: " + str(b_split))

        # Important note: (faculty, timeslot, classroom) is the order of the split triplet!!!
        if a_split[0] == b_split[0] and a_split[1] == b_split[1]:
            return False
        if a_split[1] == b_split[1] and a_split[2] == b_split[2]:
            return False

        # If no constraint violations, return true.
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
result = csp.backtracking_search(problem)
# result = csp.min_conflicts(problem, max_steps=100000)

endTime = time.time()

""" A CSP solution printer copied from csp.py. and modified for course scheduling. """


def print_solution(my_results):
    print("\nSolution: " + str(my_results))
    split_results = str(my_results).split()

    if debug:
        for split in split_results:
            print("Split solution: " + str(split))

    print("\nTime taken to find solution: " + str(endTime - startTime))


""" Print the solution (or lack thereof). """

if problem.goal_test(problem.infer_assignment()):
    print("Solution:\n")
    print_solution(result)
else:
    print("\nfailed...")
    print(problem.curr_domains)
    problem.display(problem.infer_assignment())

    print("\nTime taken to execute algorithm: " + str(endTime - startTime))

############################################################################################
############################################################################################
