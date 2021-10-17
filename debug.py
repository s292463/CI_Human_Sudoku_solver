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



simple_sudoku1 = np.array([[6, 0, 4,    0, 7, 0,    0, 0, 1],
                          [0, 5, 0,    0, 0, 0,    0, 7, 0], 
                          [7, 0, 0,    5, 9, 6,    8, 3, 4], 
       
                          [0, 8, 0,    0, 0, 2,    4, 9, 0], 
                          [1, 0, 0,    0, 0, 0,    0, 0, 3], 
                          [0, 6, 9,    7, 0, 0,    0, 5, 0], 
       
                          [9, 1, 8,    3, 6, 7,    0, 0, 5], 
                          [0, 4, 0,    0, 0, 0,    0, 6, 0], 
                          [2, 0, 0,    0, 5, 0,    7, 0, 8]], dtype=np.int8)






# %% My solution

modified = True

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


def subsetGenerator(completeSet, dimSet=None):
    if dimSet == None: dimSet = range(2, len(completeSet)) 

    for r in dimSet:
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
                                global modified 
                                if not modified: modified = True
                            except ValueError: 
                                pass

        for node in cc:
            graph.remove_node(node)


def getCellWithCandidate(collection: dict, num):
    return list(filter(lambda item : num in item[1] , collection.items()))

def onSameCol(collection:dict):
    return all(x[0][1] == collection[0][0][1] for x in collection)

def onSameRow(collection:dict):
    return all(x[0][0] == collection[0][0][0] for x in collection)


def removeFromCollection(collection, all_cellsWithCandidates: dict, numToRemove):

    for cell in collection:
        try:
            all_cellsWithCandidates[cell[0]].remove(numToRemove)
            global modified 
            if not modified: modified = True
        except ValueError: 
            pass

def findPointingTuple_or_Triple(box: dict, all_cellsWithCandidates:dict):
    # Invert the loop of subset and the number one to optimize
    
    iter =  reduce(np.union1d, (box.values())).tolist() if len(box)>1 else box.values()
    
    for i in iter:

        cells = getCellWithCandidate(box, i)

        for s in subsetGenerator(cells, dimSet = [3,2]):
            if s!=[]: # if len(s):
                if onSameCol(s):
                    reducedCol = {k:v for (k,v) in all_cellsWithCandidates.items() if k[1]==s[0][0][1] and k not in [j[0] for j in s]}
                    colCellsWithCandidates = getCellWithCandidate(reducedCol, i)

                    if len(s) == len(cells):
                        if len(colCellsWithCandidates) > 0:
                            removeFromCollection(colCellsWithCandidates, all_cellsWithCandidates, i)
                        
                    elif len(s) < len(cells):
                        reducedBox = [cell for cell in cells if cell[0] not in [c[0] for c in s]]
                        if len(colCellsWithCandidates) == 0:
                            removeFromCollection(reducedBox, all_cellsWithCandidates, i)
                        

                elif onSameRow(s):
                    reducedRow = {k:v for (k,v) in all_cellsWithCandidates.items() if k[0]==s[0][0][0] and k not in [j[0] for j in s]}
                    rowCellsWithCandidates = getCellWithCandidate(reducedRow, i)

                    if len(s) == len(cells):
                        if len(rowCellsWithCandidates) > 0:
                            removeFromCollection(rowCellsWithCandidates, all_cellsWithCandidates, i)                          

                    elif len(s) < len(cells):
                        reducedBox = [cell for cell in cells if cell[0] not in [c[0] for c in s]]
                        if len(rowCellsWithCandidates) == 0:
                            removeFromCollection(reducedBox, all_cellsWithCandidates, i)
                        


def findSingleCandidate(sudoku:np.ndarray, collection:dict, collectionType: str, all_cellsWithCandidates):
    
    if len(collection) ==0: return

    for i in list(reduce(np.union1d, (collection.values()))):
        
        cells = getCellWithCandidate(collection, i)

        # If we have only a number in the collection who is not repeating itself
        # we set the corrispondent cell of the sudoku to that number
        if len(cells) == 1:
            cell = cells[0][0]

            if collectionType =="box":
                # remove from row
                row = [(k,v) for k,v in all_cellsWithCandidates.items() if k[0] == cell[0] and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(row, all_cellsWithCandidates, i)
                # remove from col
                col = [(k,v) for k,v in all_cellsWithCandidates.items() if k[1] == cell[1] and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(col, all_cellsWithCandidates, i)

            elif collectionType =="col":
                # remove from row
                row = [(k,v) for k,v in all_cellsWithCandidates.items() if k[0] == cell[0] and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(row, all_cellsWithCandidates, i)
                # remove from col
                box = [(k,v) for k,v in all_cellsWithCandidates.items() if k[0]//3*3 == cell[0]//3*3 and k[1]//3*3 == cell[1]//3*3 and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(box, all_cellsWithCandidates, i)

            elif collectionType =="row":
                # remove from row
                #TODO: Implement helper function "getBoxNumber" given a cell
                box = [(k,v) for k,v in all_cellsWithCandidates.items() if k[0]//3*3 == cell[0]//3*3 and k[1]//3*3 == cell[1]//3*3 and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(box, all_cellsWithCandidates, i)
                # remove from col
                col = [(k,v) for k,v in all_cellsWithCandidates.items() if k[1] == cell[1] and (k,v)!= cell]#TODO: try to remove the cell disuguagliance
                removeFromCollection(col, all_cellsWithCandidates, i)


            all_cellsWithCandidates.pop(cell)
            sudoku[cell] = i

            global modified
            if not modified: modified = True

            return


def my_solver(sudoku):

    global modified
    all_cellsWithCandidates = sudoku_parser(sudoku)

    while len(all_cellsWithCandidates):
        if modified:
            modified = False
            # Every Rows
            for i in range(9):
                row = {k:v for (k,v) in all_cellsWithCandidates.items() if k[0]==i}
                findSingleCandidate(sudoku, row, "row",all_cellsWithCandidates)
                row = {k:v for (k,v) in all_cellsWithCandidates.items() if k[0]==i}
                findTuple(row, all_cellsWithCandidates)
            # Every Columns
            for i in range(9):
                col = {k:v for (k,v) in all_cellsWithCandidates.items() if k[1]==i}
                findSingleCandidate(sudoku, col, "col", all_cellsWithCandidates)
                col = {k:v for (k,v) in all_cellsWithCandidates.items() if k[1]==i}
                findTuple(col, all_cellsWithCandidates)
            # Every Boxes
            for i in range(9):
                box = {k:v for (k,v) in all_cellsWithCandidates.items() if (k[1]//3 + k[0]//3*3) == i}
                findSingleCandidate(sudoku, box, "box", all_cellsWithCandidates)
                box = {k:v for (k,v) in all_cellsWithCandidates.items() if (k[1]//3 + k[0]//3*3) == i}
                findPointingTuple_or_Triple(box, all_cellsWithCandidates)
                box = {k:v for (k,v) in all_cellsWithCandidates.items() if (k[1]//3 + k[0]//3*3) == i}
                findTuple(box, all_cellsWithCandidates)
            
            # print_sudoku(sudoku)

            print()
            
        else:
            # Make guess on the first free cell
            # Save the guess 
            # continue
            return 



















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
    print_sudoku(sudoku)
    solution = my_solver(sudoku)
    if solution is not None:
        print_sudoku(solution)


# %%



