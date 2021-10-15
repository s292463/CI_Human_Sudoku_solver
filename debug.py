# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Copyright **`(c)`** 2021 Giovanni Squillero `<squillero@polito.it>`  
# `https://github.com/squillero/computational-intelligence`  
# Free for personal or classroom use; see 'LICENCE.md' for details.

# %%
import logging
from collections import deque, Counter
from queue import PriorityQueue
import networkx as nx
from itertools import chain, combinations
import matplotlib.pyplot as plt
from pprint import pprint
from networkx.algorithms import components
from networkx.algorithms.components import connected
import numpy as np
from functools import reduce
from numpy.lib import union1d

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)


# %%
def _contains_duplicates(X):
    return np.sum(np.unique(X)) != np.sum(X)

def contains_duplicates(sol):
    return (any(_contains_duplicates(sol[r,:]) for r in range(9)) or          
            any(_contains_duplicates(sol[:,r]) for r in range(9)) or            
            any(_contains_duplicates(sol[r:r+3:,c:c+3]) for r in range(0,9,3) for c in range(0,9,3)))

def valid_solution(sol):
    return not contains_duplicates(sol) and np.sum(sol) == (1+2+3+4+5+6+7+8+9) * 9

def print_sudoku(sudoku):
    print("+-------+-------+-------+")
    for b in range(0, 9, 3):
        for r in range(3):
            print("|", " | ".join(" ".join(str(_) for _ in sudoku[b+r, c:c+3]) for c in range(0, 9, 3)), "|")
        print("+-------+-------+-------+")


# %%
def dfsolve(sudoku):
    """Vanilla depth-first solver for sudoku puzzles"""
    frontier = deque([sudoku.copy()])
    num_nodes = 0
    while frontier:
        node = frontier.popleft()
        num_nodes += 1

        if valid_solution(node):
            logging.info(f"Solved after expanding {num_nodes:,} nodes")
            return node

        for i, j in zip(*np.where(node == 0)):
            for c in range(1, 10):
                node[i, j] = c
                if not contains_duplicates(node):
                    frontier.appendleft(node.copy())
    logging.info(f"Giving up after expanding {num_nodes:,} nodes")
    return None


# %%
simple_sudoku = np.array([[0, 6, 0,    0, 5, 0,    0, 0, 3],
                          [0, 0, 0,    0, 1, 2,    0, 5, 6], 
                          [4, 0, 5,    0, 7, 6,    2, 1, 0], 
       
                          [3, 2, 7,    5, 8, 4,    0, 6, 0], 
                          [0, 4, 0,    2, 6, 7,    0, 0, 5], 
                          [8, 5, 6,    0, 3, 0,    0, 0, 2], 
       
                          [0, 9, 0,    0, 2, 0,    6, 0, 0], 
                          [0, 0, 8,    0, 4, 0,    0, 0, 0], 
                          [0, 1, 0,    0, 9, 0,    5, 0, 0]], dtype=np.int8)

# %% My solution

def buildGraph(components):
    undi_graph = nx.Graph()

    for id_comp in components:
        for other_id_comp in components:
            if(id_comp == other_id_comp): continue

            if np.intersect1d(components[id_comp], components[other_id_comp]).size != 0:
                undi_graph.add_edge(id_comp, other_id_comp)
    
    return undi_graph


def sudoku_parser(sudoku:np.ndarray):
    all_nums = [i for i in range(1,10)]
    candidates_cells={}

    for i, row in enumerate(sudoku):
        for j, num in enumerate(row):
            blockRow = i//3*3
            blockCol = j//3*3
            if num == 0:
                candidates_cells[(i,j)] = reduce(np.setdiff1d, (all_nums, row, sudoku[:,j], sudoku[blockRow:blockRow+3, blockCol:blockCol+3 ].flatten())).tolist()

    return candidates_cells


def subsetGenerator(completeSet):

    for r in range(2, len(completeSet)):
        for s in list(combinations(completeSet, r)):
            yield s


def findTuple(cellsWithCandidates, all_cellsWithCandidates):
    graph = buildGraph(cellsWithCandidates)

    while graph.number_of_nodes() != 0:

        # Find connected component in the cellGraph
        cc = list(nx.connected_components(graph))[0]

        if len(cc) > 1:

            for subset in subsetGenerator(cc):

                # TODO: verify if this condition is already satisfied in the subsetGenerator function
                if len(subset) < 2 or len(subset) == len(cc): continue 

                candidates = reduce(np.union1d, (all_cellsWithCandidates[x] for x in subset))
                
                # The lenght of subset at this point should already be minor of component nodes
                if len(candidates) == len(subset) and len(subset) < len(cc):

                    cellsWithPossibleEliminations = list(filter(lambda x:  x not in subset, cc))

                    for cell in cellsWithPossibleEliminations:
                        for candidate in candidates:
                            try:
                                all_cellsWithCandidates[cell].remove(candidate)
                            except ValueError: 
                                pass

        for node in cc:
            graph.remove_node(node)


def getCellWithCandidate(collection: dict, num):
    return list(filter(lambda item : num in item[1] , collection.items()))

def findSingleCandidate(sudoku:np.ndarray, collection:dict, all_cellsWithCandidates):

    for i in range(1, 10):
        cells = getCellWithCandidate(collection, i)
        # If we have only a number in the collection who is not repeating itself
        # we set the corrispondent cell of the sudoku to that number
        if len(cells) == 1:
            cell = cells[0][0]
            all_cellsWithCandidates.pop(cell)
            sudoku[cell[0],cell[1]] = i
            return

def my_solver(sudoku):

    all_cellsWithCandidates = sudoku_parser(sudoku)

    while len(all_cellsWithCandidates):
        # Every Rows
        for i in range(9):
            row = {k:v for (k,v) in all_cellsWithCandidates.items() if k[0]==i}
            findSingleCandidate(sudoku, row, all_cellsWithCandidates)
            findTuple(row, all_cellsWithCandidates)
        # Every Columns
        for i in range(9):
            col = {k:v for (k,v) in all_cellsWithCandidates.items() if k[1]==i}
            findSingleCandidate(sudoku, col, all_cellsWithCandidates)
            findTuple(col, all_cellsWithCandidates)
        # Every Boxes
        for i in range(9):
            box = {k:v for (k,v) in all_cellsWithCandidates.items() if (k[1]//3 + k[0]//3*3) == i}
            findSingleCandidate(sudoku, box, all_cellsWithCandidates)
            findTuple(box, all_cellsWithCandidates)
        
        print_sudoku(sudoku)

        print()


# %%
def sudoku_generator(sudokus=1, *, kappa=5, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    for puzzle in range(sudokus):
        sudoku = np.zeros((9, 9), dtype=np.int8)
        for cell in range(np.random.randint(kappa)):
            for p, val in zip(np.random.randint(0, 8, size=(9, 2)), range(1, 10)):
                tmp = sudoku.copy()
                sudoku[tuple(p)] = val
                if contains_duplicates(sudoku):
                    sudoku = tmp
        yield sudoku.copy()


# %%
for sudoku in sudoku_generator(random_seed=42):
    print_sudoku(simple_sudoku)
    solution = my_solver(simple_sudoku)
    if solution is not None:
        print_sudoku(solution)


# %%



