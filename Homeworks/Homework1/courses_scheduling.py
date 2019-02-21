"""
This uses a pre-built problem specification for the
Zebra puzzle taken from AIMA's csp.py. See:

    https://en.wikipedia.org/wiki/Zebra_Puzzle

@author: kvlinden
@version 14feb2013
"""
import time

from csp import Zebra, min_conflicts, backtracking_search, AC3
from search import depth_first_graph_search


def print_solution(result):
    """A CSP solution printer copied from csp.py."""
    for h in range(1, 6):
        print('House', h)
        for (var, val) in result.items():
            if val == h: print('\t', var)


puzzle = Zebra()

# result = depth_first_graph_search(puzzle)
# result = AC3(puzzle)
result = backtracking_search(puzzle)
# result = min_conflicts(puzzle, max_steps=1000)

if puzzle.goal_test(puzzle.infer_assignment()):
    print("Solution:\n")
    print_solution(result)
else:
    print("failed...")
    print(puzzle.curr_domains)
    puzzle.display(puzzle.infer_assignment())

# The Zebra Puzzle



# def Zebra():
#     """Return an instance of the Zebra Puzzle."""
#     Colors = 'Red Yellow Blue Green Ivory'.split()
#     Pets = 'Dog Fox Snails Horse Zebra'.split()
#     Drinks = 'OJ Tea Coffee Milk Water'.split()
#     Countries = 'Englishman Spaniard Norwegian Ukranian Japanese'.split()
#     Smokes = 'Kools Chesterfields Winston LuckyStrike Parliaments'.split()
#     variables = Colors + Pets + Drinks + Countries + Smokes
#     domains = {}
#     for var in variables:
#         domains[var] = list(range(1, 6))
#     domains['Norwegian'] = [1]
#     domains['Milk'] = [3]
#     neighbors = parse_neighbors("""Englishman: Red;
#                 Spaniard: Dog; Kools: Yellow; Chesterfields: Fox;
#                 Norwegian: Blue; Winston: Snails; LuckyStrike: OJ;
#                 Ukranian: Tea; Japanese: Parliaments; Kools: Horse;
#                 Coffee: Green; Green: Ivory""", variables)
#     for type in [Colors, Pets, Drinks, Countries, Smokes]:
#         for A in type:
#             for B in type:
#                 if A != B:
#                     if B not in neighbors[A]:
#                         neighbors[A].append(B)
#                     if A not in neighbors[B]:
#                         neighbors[B].append(A)
#
#     def zebra_constraint(A, a, B, b, recurse=0):
#         same = (a == b)
#         next_to = abs(a - b) == 1
#         if A == 'Englishman' and B == 'Red':
#             return same
#         if A == 'Spaniard' and B == 'Dog':
#             return same
#         if A == 'Chesterfields' and B == 'Fox':
#             return next_to
#         if A == 'Norwegian' and B == 'Blue':
#             return next_to
#         if A == 'Kools' and B == 'Yellow':
#             return same
#         if A == 'Winston' and B == 'Snails':
#             return same
#         if A == 'LuckyStrike' and B == 'OJ':
#             return same
#         if A == 'Ukranian' and B == 'Tea':
#             return same
#         if A == 'Japanese' and B == 'Parliaments':
#             return same
#         if A == 'Kools' and B == 'Horse':
#             return next_to
#         if A == 'Coffee' and B == 'Green':
#             return same
#         if A == 'Green' and B == 'Ivory':
#             return a - 1 == b
#         if recurse == 0:
#             return zebra_constraint(B, b, A, a, 1)
#         if ((A in Colors and B in Colors) or
#                 (A in Pets and B in Pets) or
#                 (A in Drinks and B in Drinks) or
#                 (A in Countries and B in Countries) or
#                 (A in Smokes and B in Smokes)):
#             return not same
#         raise Exception('error')
#     return CSP(variables, domains, neighbors, zebra_constraint)


# def solve_zebra(algorithm=min_conflicts, **args):
#     z = Zebra()
#     ans = algorithm(z, **args)
#     for h in range(1, 6):
#         print('House', h, end=' ')
#         for (var, val) in ans.items():
#             if val == h:
#                 print(var, end=' ')
#         print()
#     return ans['Zebra'], ans['Water'], z.nassigns, ans