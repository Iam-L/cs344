"""
Course: CS 344 - Artificial Intelligence
Instructor: Professor VanderLinden
Name: Joseph Jinn
Date: 2-20-19

Homework 1 - Problem Solving
Course Scheduling Constraint Satisfaction Problem

Note to self:

TODO - Ask Professor VanderLinden about the parts I don't quite understand.

FIXME - Not working at the moment.
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

# Define a course scheduling formulation.
def courses():
    # Defines the variables.
    Courses = 'cs108 cs112 cs212 cs214 cs232 cs262 cs344'.split()
    Faculty = 'adams vanderlinden plantinga wieringa norman'.split()
    TimeSlots = 'mwf8:00-8:50 mwf9:00-9:50 mwf10:30-11:20 tth11:30-12:20 tth12:30-1:20'.split()
    Classrooms = 'nh253 sb382'.split()
    variables = Courses + Faculty + TimeSlots + Classrooms

    # Defines the domain.
    domains = {}
    for var in variables:
        domains[var] = list(range(1, 7))
    domains['cs108'] = [1]
    domains['cs112'] = [2]
    domains['cs212'] = [3]
    domains['cs214'] = [4]
    domains['cs232'] = [5]
    domains['cs262'] = [6]
    domains['cs344'] = [7]

    # Debug.
    for d in domains:
        print("Domain value: " + d)

    # Define the values associated with variables.
    neighbors = csp.parse_neighbors("""adams: cs112;
                adams: cs214; vanderlinden: cs344; wieringa: cs108;
                plantinga: cs212; norman: cs232; vanderlinden: cs262""", variables)

    # Debug.
    for n in neighbors:
        print("Neighbors Value:" + n)

    for types in [Courses, Faculty, TimeSlots, Classrooms]:
        for A in types:
            for B in types:
                if A != B:
                    if B not in neighbors[A]:
                        neighbors[A].append(B)
                    if A not in neighbors[B]:
                        neighbors[B].append(A)

    # Define the constraints on the variables.
    def scheduling_constraint(A, a, B, b, recurse=0):
        same = (a == b)
        next_to = abs(a - b) == 1
        if A == 'adams' and B == 'cs112':
            return same
        if A == 'adams' and B == 'cs214':
            return same
        if A == 'vanderlinden' and B == 'cs344':
            return same
        if A == 'wieringa' and B == 'cs108':
            return same
        if A == 'plantinga' and B == 'cs212':
            return same
        if A == 'norman' and B == 'cs232':
            return same
        if A == 'vanderlinden' and B == 'cs262':
            return same
        if recurse == 0:
            return scheduling_constraint(B, b, A, a, 1)
        if ((A in Courses and B in Courses) or
                (A in Faculty and B in Faculty) or
                (A in TimeSlots and B in TimeSlots) or
                (A in Classrooms and B in Classrooms)):
            return not same
        raise Exception('error')

    return csp.CSP(variables, domains, neighbors, scheduling_constraint)


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


def print_solution(result):
    for h in range(1, 6):
        print('House', h)
        for (var, val) in result.items():
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

############################################################################################
############################################################################################
