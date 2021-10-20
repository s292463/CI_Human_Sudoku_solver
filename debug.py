# %%
import logging
from copy import deepcopy
from collections import deque

import networkx as nx
from itertools import combinations
import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from time import time

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

simple_sudoku2 = np.array([
                          [0, 0, 5,    0, 0, 0,    6, 0, 0],
                          [0, 4, 0,    6, 0, 0,    0, 2, 0], 
                          [3, 0, 0,    0, 7, 0,    0, 0, 5], 
       
                          [0, 2, 0,    0, 0, 6,    0, 0, 0], 
                          [0, 0, 1,    0, 0, 0,    3, 0, 0], 
                          [0, 0, 0,    4, 0, 0,    0, 6, 0], 
       
                          [5, 0, 0,    0, 3, 0,    0, 0, 1], 
                          [0, 9, 0,    0, 0, 2,    0, 4, 0], 
                          [0, 0, 8,    0, 0, 0,    5, 0, 0]], dtype=np.int8)

simple_sudoku3 = np.array([
                          [4, 3, 9,    0, 0, 0,    0, 1, 0],
                          [0, 0, 0,    4, 9, 0,    0, 8, 0], 
                          [6, 1, 0,    3, 0, 8,    2, 5, 0], 
       
                          [0, 9, 5,    8, 0, 0,    4, 0, 0], 
                          [0, 0, 7,    0, 0, 0,    0, 6, 0], 
                          [0, 0, 0,    0, 0, 1,    5, 0, 0], 
       
                          [0, 0, 0,    0, 0, 7,    0, 0, 0], 
                          [0, 0, 0,    0, 8, 2,    7, 0, 0], 
                          [0, 0, 0,    0, 0, 0,    0, 0, 0]], dtype=np.int8)

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

def getBoxNumber(cell):
    return cell[0]//3 + cell[1]//3 * 3

def sudoku_parser(sudoku):
    all_nums = list(range(1, 10))
    candidates_cells = dict()
    rows = dict()
    cols = dict()
    boxes = dict()

    for i, row in enumerate(sudoku):
        for j, num in enumerate(row):
            blockRow = i//3*3
            blockCol = j//3*3
            if num == 0:
                candidates_cells[(i,j)] = reduce(np.setdiff1d, (all_nums, row, sudoku[:,j], sudoku[blockRow : blockRow + 3, blockCol : blockCol + 3].flatten())).tolist()
    for i in range(9):
        rows[i] = {k:v for k,v in candidates_cells.items() if k[0] == i}
        cols[i] = {k:v for k,v in candidates_cells.items() if k[1] == i}
        boxes[i] = {k:v for k,v in candidates_cells.items() if getBoxNumber(k) == i}

    return candidates_cells, rows, cols, boxes

def subsetGenerator2(n, k):
    a = np.ones((k, n-k+1), dtype=int)
    a[0] = np.arange(n-k+1)
    for j in range(1, k):
        reps = (n-k+j) - a[j-1]
        a = np.repeat(a, reps, axis=1)
        ind = np.add.accumulate(reps)
        a[j, ind[:-1]] = 1-reps[1:]
        a[j, 0] = j
        a[j] = np.add.accumulate(a[j])
    return a

def subsetGenerator(completeSet, dimSet=None):
    if dimSet == None: dimSet = range(2, len(completeSet)) 

    for r in dimSet:
        for s in list(combinations(completeSet, r)):
            yield s

def findTuple(cellsWithCandidates, all_cellsWithCandidates, counterOfModifications = None):
    graph = buildGraph(cellsWithCandidates)

    while graph.number_of_nodes() != 0:

        # Find connected component in the cellGraph and take the first one cc
        cc = list(nx.connected_components(graph))[0]

        if len(cc) > 1:

            for subset in subsetGenerator(cc):

                candidates = reduce(np.union1d, (cellsWithCandidates[x] for x in subset))
                
                if len(candidates) == len(subset):

                    cellsWithPossibleEliminations = list(filter(lambda x:  x not in subset, cc))

                    for candidate in candidates:
                        removeFromCollection(cellsWithPossibleEliminations, all_cellsWithCandidates, candidate, counterOfModifications)

        for node in cc:
            graph.remove_node(node)

def getCellWithCandidate(collection, all_cellsWithCandidates, num):
    return list(filter(lambda item : num in all_cellsWithCandidates[item] , collection))

def onSameCol(collection):
    return all(x[1] == collection[0][1] for x in collection)

def onSameRow(collection):
    return all(x[0] == collection[0][0] for x in collection)

def removeFromCollection(collection, all_cellsWithCandidates, numToRemove, counterOfModifications = None):
    for cell in collection:
        try:
            all_cellsWithCandidates[cell].remove(numToRemove)
            if counterOfModifications != None: 
                counterOfModifications[0] += 1
            global modified 
            if not modified: modified = True
        except ValueError: 
            pass

def popFromStruct(all_cellsWithCandidates, rows, cols, boxes, cell):
    all_cellsWithCandidates.pop(cell)
    rows[cell[0]].pop(cell)
    cols[cell[1]].pop(cell)
    boxes[getBoxNumber(cell)].pop(cell)

def addToSudoku(sudoku, cell, num, counterOfModifications = None):
    global modified
    
    sudoku[cell] = num 
    if counterOfModifications!= None: counterOfModifications[0] += 1
    if not modified: modified = True

def findPointingTuple_or_Triple(box, all_cellsWithCandidates, counterOfModifications):
    # Invert the loop of subset and the number one to optimize
    
    iter =  reduce(np.union1d, box.values()).tolist() if len(box)>1 else box.values()
    
    for i in iter:

        cells = getCellWithCandidate(box.keys(), all_cellsWithCandidates, i)
        # for s in subsetGenerator2(cells, 3):
        for s in subsetGenerator(cells, dimSet = [3,2]):
            if len(s):
                if onSameCol(s):
                    reducedCol = [k for k in all_cellsWithCandidates.keys() if k[1] == s[0][1] and k not in s]

                    if len(s) == len(cells):
                        if len(reducedCol) > 0:
                            removeFromCollection(reducedCol, all_cellsWithCandidates, i, counterOfModifications)
                        
                    elif len(s) < len(cells):
                        reducedBox = [cell for cell in cells if cell[0] not in [c[0] for c in s]]
                        if len(reducedCol) == 0:
                            removeFromCollection(reducedBox, all_cellsWithCandidates, i, counterOfModifications)
                        

                elif onSameRow(s):
                    reducedRow = [k for k in all_cellsWithCandidates.keys() if k[0] == s[0][0] and k not in s]

                    if len(s) == len(cells):
                        if len(reducedRow) > 0:
                            removeFromCollection(reducedRow, all_cellsWithCandidates, i, counterOfModifications)                          

                    elif len(s) < len(cells):
                        reducedBox = [cell for cell in cells if cell[0] not in [c[0] for c in s]]
                        if len(reducedRow) == 0:
                            removeFromCollection(reducedBox, all_cellsWithCandidates, i, counterOfModifications)
                        
def findSingleCandidate(sudoku, collection, collectionType, all_cellsWithCandidates, counterOfModifications, rows, cols, boxes):
    
    if not len(collection): 
        return

    for i in list(reduce(np.union1d, (collection.values()))):
        cells = getCellWithCandidate(collection, all_cellsWithCandidates, i)

        # If we have only a number in the collection who is not repeating itself
        # we set the corrispondent cell of the sudoku to that number
        if len(cells) == 1:
            cell = cells[0]

            if collectionType == "box":
                # remove from row
                row = rows[cell[0]]
                removeFromCollection(row, all_cellsWithCandidates, i, counterOfModifications)
                # remove from col
                col = cols[cell[1]]
                removeFromCollection(col, all_cellsWithCandidates, i, counterOfModifications)

            elif collectionType == "col":
                # remove from row
                row = rows[cell[0]]
                removeFromCollection(row, all_cellsWithCandidates, i, counterOfModifications)
                # remove from box
                box = boxes[getBoxNumber(cell)]
                removeFromCollection(box, all_cellsWithCandidates, i, counterOfModifications)

            elif collectionType == "row":
                # remove from box
                box = boxes[getBoxNumber(cell)]
                removeFromCollection(box, all_cellsWithCandidates, i, counterOfModifications)
                # remove from col
                col = cols[cell[1]]
                removeFromCollection(col, all_cellsWithCandidates, i, counterOfModifications)

            # Remove from the data and add the number to the sudoku 
            popFromStruct(all_cellsWithCandidates, rows, cols, boxes, cell)
            addToSudoku(sudoku, cell, i, counterOfModifications)
            
            return

def my_solver(sudoku):
    # TODO: parallelize the tasks in boxes, cols and rows tasks

    # Used to track if the algorithm modifies the sudoku's structure
    global modified
    wrongGuess = False

    expandedNodes = 0
    algorithmExecutions = 0
    rowModifications = [0]
    colModifications = [0]
    boxModifications = [0]

    states = deque()
    all_cellsWithCandidates, rows, cols, boxes = sudoku_parser(sudoku)

    start = time()

    while len(all_cellsWithCandidates):
        algorithmExecutions += 1

        if list() in all_cellsWithCandidates.values(): wrongGuess = True

        if modified and not wrongGuess:
            modified = False
            
            for i in range(9):

                # Every Boxes
                box = boxes[i]
                findPointingTuple_or_Triple(box, all_cellsWithCandidates, boxModifications)
                findTuple(box, all_cellsWithCandidates, boxModifications)
                findSingleCandidate(sudoku, box, "box", all_cellsWithCandidates, boxModifications, rows, cols, boxes)
                # Every Rows
                row = rows[i]
                findTuple(row, all_cellsWithCandidates, rowModifications)
                findSingleCandidate(sudoku, row, "row", all_cellsWithCandidates, rowModifications, rows, cols, boxes)
                # Every Columns
                col = cols[i]
                findTuple(col, all_cellsWithCandidates, colModifications)
                findSingleCandidate(sudoku, col, "col", all_cellsWithCandidates, colModifications, rows, cols, boxes)
                
            
        else:
            parsedCandidates = None
            
            # State recovery
            if(wrongGuess):
                wrongGuess = False

                # No solution Case
                if len(states) == 0: return None, (), ()

                all_cellsWithCandidates, rows, cols, boxes, parsedCandidates = states.pop()
                while parsedCandidates == []:
                    all_cellsWithCandidates, rows, cols, boxes, parsedCandidates = states.pop()
           
            # Take the first empty cell
            myGuessCell = list(all_cellsWithCandidates.keys())[0]
            # Take the first candidate
            myGuessCandidates = all_cellsWithCandidates[myGuessCell] if not parsedCandidates else parsedCandidates
            myGuessCandidate = myGuessCandidates[0]
            # Save the state
            states.append((deepcopy(all_cellsWithCandidates), deepcopy(rows), deepcopy(cols), deepcopy(boxes), myGuessCandidates[1:-1]))
            # Remove the cell from the cells with candidates dictionary
            popFromStruct(all_cellsWithCandidates, rows, cols, boxes, myGuessCell)
            # Put the candidate inside the sudoku
            addToSudoku(sudoku, myGuessCell, myGuessCandidate)
            # Expand a node every guess
            expandedNodes += 1
            
            # Delete the candidate from the row/col/box of the chosen cell
            rowColBox = {**rows[myGuessCell[0]], **cols[myGuessCell[1]], **boxes[getBoxNumber(myGuessCell)]}
            removeFromCollection(rowColBox, all_cellsWithCandidates, myGuessCandidate)
 
    executionTime = time() - start

    if valid_solution(sudoku):
        print(f"Solved in {executionTime}")
        print(f"Valid solution found with {expandedNodes} expanded nodes")
        print(f"Algorithm excecuted {algorithmExecutions} times")
        print(f"Rows section make modifcations {rowModifications} times")
        print(f"Columns section make modifcations {colModifications} times")
        print(f"Boxes section make modifcations {boxModifications} times")

    return sudoku, (expandedNodes, algorithmExecutions, executionTime), (rowModifications, colModifications, boxModifications)

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

def printBarChart(x, y, chart, label):
    chart.bar(x, y, label=label)
    chart.set_xlabel('Sudokus')
    chart.legend(loc='upper right')


# %%
rowsAccumulator = list()
colsAccumulator = list()
boxesAccumulator = list()
executionTimeAccumulator = list()
expandedNodesAccumulator = list()
algorithmExecutionsAccumulator = list()

numSudoku = solvedSudokus = 40

for sudoku in sudoku_generator(sudokus = numSudoku, random_seed=42):
    print_sudoku(sudoku)
    solution, generalPerformances, sectionPerformances = my_solver(sudoku)
     
    if solution is not None:
        print_sudoku(solution)
        
        expandedNodesAccumulator.append(generalPerformances[0])
        algorithmExecutionsAccumulator.append(generalPerformances[1])
        executionTimeAccumulator.append(generalPerformances[2])

        rowsAccumulator.append(sectionPerformances[0][0])
        colsAccumulator.append(sectionPerformances[1][0])
        boxesAccumulator.append(sectionPerformances[2][0])

        print("\n\n")
    else:
        print("No solution")
        solvedSudokus -= 1
    
labels = np.array([['Rows', 'Cols', 'Boxes'],['Number of expanded nodes', 'Number of Algorithm executions', 'Executions Time']])
data = np.array([[rowsAccumulator, colsAccumulator, boxesAccumulator], [expandedNodesAccumulator,algorithmExecutionsAccumulator,executionTimeAccumulator]])

fig, ax = plt.subplots(2, 3, figsize=(20,10))
for i, row in enumerate(ax):
    for j, chart in enumerate(row):
        printBarChart(range(solvedSudokus), data[i,j], chart, labels[i,j])

plt.show()

