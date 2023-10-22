# HW_2 Sandbox
# IMPORT FUNCTIONS
import timeit
from tqdm import tqdm
import numpy as np
import heapq as h
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pdb
import math
import math as m
#from icecream import ic
import pandas as pd
import random
import time
import threading
import datetime
import pickle
import os
import concurrent
from concurrent.futures import ThreadPoolExecutor


# NODE CLASS
class Function_node(object):
    # Global parameters tha bound acceptable operators
    # last is the header indicator (indexing from 1!)
    node_types = ('+', '-', '*', '/', '^', 'sin', 'cos',
                  'const', 'X', 'ERR')
    dual_operators = ('+', '-', '*', '/', '^')
    single_operators = ('sin', 'cos')

    # Constructor for a function node
    def __init__(self, function_name, value=None):
        self.num_childs = 0
        self.value = value
        # assigning a name in our accepted list
        try:
            if function_name not in self.node_types:
                raise ValueError(
                    F"Invalid input: {function_name}. Please enter a valid node type \n\n\t run> heap.node_types for the list")
            else:
                self.function_name = function_name
        except ValueError as err_msg:
            # print(err_msg)
            self.function_name = 'ERROR'
            sys.exit()  # exit if there is an issue

        # set number of children gien type

        if self.function_name in self.dual_operators:
            self.req_num_childs = 2
        elif self.function_name in self.single_operators:
            self.req_num_childs = 1
        else:
            self.req_num_childs = 0

    # Node methods to integrate into our tree:
    def add_child_count(self):
        self.num_childs += 1

    def can_add_child(self):
        return self.num_childs < self.req_num_childs

    def copy_node(self):
        # returns a copy of the node
        new_node = Function_node(function_name=self.function_name,
                                 value=self.value)
        new_node.num_childs = self.num_childs
        return new_node

    # printing a node:
    def __str__(self):  # print statement
        if self.function_name == 'const':
            return f"{round(self.value,2)}"
        elif self.function_name == 'X':
            return "X"
        else:
            return f"{self.function_name}"


# ***  NP_HEAP CLASS ***
# inspiration: https://github.com/swacad/numpy-heap/blob/master/heap.py
# Starting at index 1~
class NP_Heap(Function_node):
    def __init__(self, length=2, Heap=None, Randomize=False, max_depth=4, const_prob=.45, C_range=(-10, 10)):
        self.heap = np.full(length, None, dtype=object)
        # ALL Operators ('+', '-', '*', '/', 'sin', 'cos','^',
        self.operators = ('*', '+', '-', '/', 'sin', 'cos')
        self.trig_operators = ('sin', 'cos')
        self.non_operator = ('X', 'const')
        self.MSE = None
        self.fitness = None

        if Randomize:
            self.Random_Heap(max_depth=max_depth,
                             const_prob=const_prob,
                             C_range=C_range)

 ############# Heap Manipulation #############

    # INDEXING AT 1:

    def get_parent_idx(self, child_idx):
        return int(child_idx/2)

    def get_parent(self, child_idx):
        parent_idx = self.get_parent_idx(child_idx)
        return self.heap[parent_idx]

    def get_left_child_idx(self, parent_idx):
        return 2 * parent_idx

    def get_right_child_idx(self, parent_idx):
        return 2 * parent_idx + 1

    def get_left_child(self, parent_idx):
        left_child = self.heap[self.get_left_child_idx(parent_idx)]
        return left_child

    def get_right_child(self, parent_idx):
        right_child = self.heap[self.get_right_child_idx(parent_idx)]
        return right_child

    def get_children_idx(self, parent_idx):
        return 2*parent_idx, 2*parent_idx+1

    def get_children(self, parent_idx):
        return self.heap[2*parent_idx], self.heap[2*parent_idx+1]

    def get_children_type(self, parent_idx):
        L, R = self.get_children(parent_idx)
        return type(L), type(R)

    def depth(self):
        deepest_node_ind = max(np.arange(self.heap.size)[self.heap != None])
        return int(m.floor(m.log2(deepest_node_ind)))

    def has_root(self):
        if self.heap[0] == None:
            return False
        else:
            return True

    def copy(self, given_len=None):  # TODO Test
        # ERROR HERE! WE had to not just initialize a new heap but also the nodes themselves... duh....
        length = given_len if given_len else self.heap.size
        h = NP_Heap(length=length)

        for i, node in enumerate(self.heap):
            if node == None:
                h.heap[i] = None
            else:
                h.heap[i] = node.copy_node()
        return h

 # INSERT FUNCTION
    def insert(self, parent_indx, node_obj, position=None):
        # check for size availibility or resize

        # check heap length if short, doubles array until acceptable
        while self.heap.size - 1 < self.get_right_child_idx(parent_indx):
            #print('doubled',self.heap.size ,self.get_right_child_idx(parent_indx)  )
            self.heap = np.append(self.heap,
                                  np.full(self.heap.size, None).astype(Function_node))

        L_indx, R_indx = self.get_children_idx(parent_indx)

        # if a position is provided it will insert WITH REPLACEMENT
        if position:
            if position == 'L':
                self.heap[L_indx] = node_obj
            elif position == 'R':
                self.heap[R_indx] = node_obj
            else:
                print("invalid arg position = 'L' or 'R'")

        # if no position provided it will insert left to right if a child is empty
        elif not self.heap[L_indx]:
            self.heap[L_indx] = node_obj
        elif not self.heap[R_indx]:
            self.heap[R_indx] = node_obj

        else:  # insert to the left child. #TODO: implement recursive insert.
            print('ERROR: Parent children filled')
            #self.insert(self,parent_indx = L, value = value )
        return None

 # ---------------------------------------------

 ####### DISPLAY FUNCTIONS ##########
    def __str__(self):
        return self.print_heap_tree()

    def print_heap_tree(self, index=1, prefix="", is_left=True):  # CALLED BY __str__ method
        output = ""
        if index < len(self.heap) and self.heap[index] is not None:
            # Process right child first (going down)
            output += self.print_heap_tree(index=2*index+1, prefix=prefix + (
                "|   " if is_left else "    "), is_left=False)
            # Add the current node to the output
            output += prefix + "|-- " + str(self.heap[index]) + "\n"
            # Process left child
            output += self.print_heap_tree(index=2*index, prefix=prefix + (
                "|   " if not is_left else "    "), is_left=True)
        # else:
            # Indicate the absence of a node with [...]
            # output += prefix + "|-- [...]\n"
        return output

    def print_arr(self):  # shows the heap and indexes
        heap_str = [str(node) for node in self.heap]
        ind_arr = np.arange(self.heap.size)
        max_def = max(ind_arr[self.heap != None])+1
        print(np.stack((ind_arr[1:max_def], heap_str[1:max_def])))
        return None

    def plot_approximation(self, X_range=(0, 10), target_data=None, y_pred=None,
                           data_name='Data'):
        X_arr = np.linspace(X_range[0], X_range[1], 1000)

        if type(target_data) != None:
            plt.plot(target_data[:, 0], target_data[:, 1], label='Given_data')

        if not y_pred:
            y_pred = [self.evaluate(X=x) for x in X_arr]

        plt.plot(X_arr, y_pred, label='GA Solution')
        plt.legend()
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(data_name + ' approximation ')
        plt.show()
        return None

    def build_function(self, index=1):
        # CGPT Func
        heap_array = self.heap
        # Base case: if the index is out of bounds or the node is None, return an empty string
        if index >= len(heap_array) or heap_array[index] is None:
            return ""

        # Get the current node's value
        current_node = str(heap_array[index])

        # Define operators with higher precedence
        higher_precedence = ['*', '/', 'sin', 'cos']

        # Get the left and right child indices
        left_child_index = 2 * index
        right_child_index = 2 * index + 1

        # Recursively build the left and right subtrees
        left_subtree = self.build_function(index=left_child_index)
        right_subtree = self.build_function(index=right_child_index)

        # Check if parentheses are needed around children
        if left_subtree and current_node in higher_precedence:
            left_subtree = f"({left_subtree})"
        if right_subtree and current_node in higher_precedence:
            right_subtree = f"({right_subtree})"

        # Combine the current node, left subtree, and right subtree to form the expression
        expression = f"{left_subtree}{current_node}{right_subtree}" if left_subtree or right_subtree else current_node

        return expression
 # ---------------------------------------------

 ####### Main functionalities ###########
    def Random_Heap(self, Index=1, max_depth=4, const_prob=.4, C_range=(-10, 10)):
        '''
        this is a function that will be called recursively to build valid trees to every index
            - Start at a given index (default is root = 1) initialized as an operator
                - If not initialized it assigns a random operator. 
            - given operator type, select the number of required children (can = req chlds > chlds)
            for each:
                - Check for termination criteria that make the children of that node constants
                    --> either some random probability or a maximum depth is reached
                - if not terminated, select a random operator
                insert that node left to right into the parent children 
                - note that the node was added so the can add child is accurate. 

        '''
        # initialize root.
        if self.heap[Index] == None:
            self.heap[Index] = Function_node(np.random.choice(self.operators))
        while self.heap[Index].can_add_child():  # TODO TEST

            # SORRY FOR THE TRIPPLE NESTING :(

            # criteria for a constant child
            if self.depth() >= max_depth - 1 or np.random.rand() < const_prob:  # no operators terminate with constants

                if self.heap[Index].function_name in self.trig_operators:
                    node_name = 'X'
                    node_val = None
                # check left child has been populated to ensure they are not both same (irrelevant node)
                elif self.heap[Index].num_childs == 1:
                    L_name = self.heap[2 * Index].function_name
                    # if this is the second constant make it opposit
                    if L_name in ['X', 'const']:
                        node_name = 'const' if L_name == 'X' else 'X'
                        node_val = np.random.uniform(C_range[0], C_range[1]
                                                     ) if node_name == 'const' else None
                    else:  # otherwise random
                        node_name = np.random.choice(['X', 'const'])
                        node_val = np.random.uniform(C_range[0], C_range[1]
                                                     ) if node_name == 'const' else None

                # selcting first child:.
                else:
                    node_name = np.random.choice(['X', 'const'])
                    node_val = np.random.uniform(
                        C_range[0], C_range[1]) if node_name == 'const' else None

            # operator child:
            else:
                node_name = np.random.choice(self.operators)
                node_val = None

            new_node = Function_node(function_name=node_name, value=node_val)
            self.insert(parent_indx=Index, node_obj=new_node)
            # note the addition of a child to break while loop
            self.heap[Index].add_child_count()
        # TODO maybe immediately prune/consolidate shity nodes

        L_child, R_child = self.get_children(Index)
        # self.print_arr()
        if L_child.function_name in self.operators:
            self.Random_Heap(Index=2*Index, max_depth=max_depth,
                             const_prob=const_prob, C_range=C_range)
        if type(R_child) == Function_node and R_child.function_name in self.operators:
            self.Random_Heap(2*Index+1, max_depth=max_depth,
                             const_prob=const_prob, C_range=C_range)

    # ****  EVALUATE A NODE ***:
    def evaluate(self, node_ind=1, X=1):  # tree root = 1
        # evaluates a node given its index

        def node_operation(operator, operand):
            if operator == '+':
                return operand[0]+operand[1]
            elif operator == '-':
                return operand[0] - operand[1]
            elif operator == '*':
                return operand[0] * operand[1]
            elif operator == '/':
                if operand[1] == 0:
                    return 1.0e6
                return operand[0] / operand[1]
            elif operator == '^':
                return operand[0] ** operand[1]
            elif operator == 'sin':
                return np.sin(operand)
            elif operator == 'cos':
                return np.cos(operand)

        # MAIN LOOP:

        L_child, R_child = self.get_children(node_ind)
        L_indx, R_indx = self.get_children_idx(node_ind)
        children_types = self.get_children_type(node_ind)

        # CHECKS left child, if an operator, evaluate recursively returning a constant. If X, assign it.

        # checks for None leaf
        if type(L_child) is None:
            pass

        elif L_child.function_name in self.operators:
            L_child.value = self.evaluate(node_ind=2*node_ind, X=X)

        elif L_child.function_name == 'X':
            L_child.value = X
        if R_child == None:
            pass
        elif R_child.function_name in self.operators:
            R_child.value = self.evaluate(node_ind=2*node_ind+1, X=X)

        elif R_child.function_name == 'X':
            R_child.value = X

        # terminating state: both children are constandts (floats) or Nan (with at least a constant) after being evaluated
        node_operator = self.heap[node_ind].function_name
        try:
            if Function_node not in children_types:
                raise TypeError(
                    f"Invalid children type for operator: {node_operator} \n\t L/R children are: {(L_child, R_child)}")

            # i.e its sin, cos, tan etc (as defined above
            elif node_operator in self.trig_operators:

                if None not in (L_child, R_child):
                    raise ValueError(
                        f"Invalid children type for operator: {node_operator} \n\t L/R children are: {(L_child, R_child)}")
                elif type(L_child):  # if None use the right child value
                    node_val = node_operation(node_operator, L_child.value)
                elif R_child:
                    node_val = node_operation(node_operator, R_child.value)
            else:
                node_val = node_operation(
                    node_operator, (L_child.value, R_child.value))

        except ValueError as err_msg:
            sys.exit()

        # DIAGNOSTIC
        '''
        msg_out = (
            f"at the root evaluating with:\n\t - parent node index {node_ind}"
            f"\n\t - operator {node_operator}"
            f"\n\t - with children {[str(x) for x in self.heap[[L_indx, R_indx]]]}"
            f"\n result passed {node_val}"
        )
        #print(msg_out)
        '''
        return node_val
    # **** END NODE EVAL   ***

    def get_MSE(self, point_cloud, plotting=False, T=.05):
        # RECALL: MSE = (1/n) * Σ(actual – forecast) ^2
        # TODO subsection (skip every other point)
        X_arr = point_cloud[:, 0]
        y = point_cloud[:, 1]
        y_pred = np.array([self.evaluate(X=x) for x in X_arr])
        self.MSE = np.sum(np.square(y_pred-y)/y.shape[0])
        self.fitness = np.exp(-T*self.MSE) + 1e-6
        self.y_pred = y_pred  # helpful for difference calcs
        # TODO Self.ypred = y_pred to store at heap
        if plotting:
            self.plot_approximation(X_arr, y_pred, y_true=y)
        return self.MSE

    def get_fitness(self, target_data, T=.05):
        self.get_MSE(target_data, T=T)
        #self.fitness = np.exp(-T*MSE) + 1e-6
        return self.fitness

 # #### EP Functions ########## :
    '''
        IDEAS: 
            - Most vanilla: search through heap for the constants and +_ by X small % of the value
            - Adding operations: for chain terminating variables, replace them with the + operator 
                                 and add a small constant to the variable
            - Mutation schedules: annealing or adaptive mutation (increases if stagnating decreases if progress)
            - Optimization by consolidating into one function so we dont have to traverse heap. (in practice heap should be small!)
    '''

    # TODO newar zero increase mutation size
    def Constant_Mutation(self, change_prcnt=.05, min_mutation=.25):
        # small change to constants and addition/substraction to variables
        # basic: for node in heap, if name = const +_ X%
        for node in self.heap:
            if type(node) == Function_node and node.function_name == 'const':
                R = abs(change_prcnt*node.value)
                # makes sure the change is at least a min for near zero constant
                Range = max(R, min_mutation)
                node.value += random.uniform(-1*Range, Range)
        return self

    def Operator_Mutation(self, swap_num=1):
        for i, node in enumerate(self.heap):
            if type(node) == Function_node:  # only operate over nodes....
                if node.function_name == '*':
                    # self.print_arr()
                    node.function_name = '/'
                    # self.print_arr()
        return None

    ##### BLOAT PROBLEM ####
    # TODO  self.depth() and getting nodes at a maximum depth
    def depth_consolidation(self, max_depth=5):
        if self.depth() > max_depth:
            for node in self.heap[max_depth]:
                # calculate averave value and consolidate the node to this value. conditional for large stdv?
                print('Need to implement')
        return None

    def pruning(self):  # TODO
        # deletes useless tree (tree that evaluates to 1 for a mult child or a 0 for a + child/sin cos delete)
        return None

    def hoist_mutation(self):  # TODO
        # Chooses a tree that evaluates to the same constant through x-range and replaces it by the constant
        return None

    # TODO
    def Random_subtree(self):
        # adds random subtree to some function
        pass

    def subtree(self, indx):
        subtree_ind = [indx]
        subtree_depth = self.depth() - int(m.floor(m.log2(indx)))  # heap depth - node depth
        for i in range(1, subtree_depth+1):
            for j in range(indx*2**i, (indx+1)*2**i):
                subtree_ind.append(j)

        return subtree_ind, self.heap[subtree_ind]


############### RANDOM SEARCH ###########################
###### RANDOM & HC ###############
def Random_Search(evaluations, data, max_depth=3, const_prob=.5, C_range=(-10, 10)):
    MSE_log = []
    best_solution = None

    best_function_err = 1e9
    for i in tqdm(range(evaluations), desc='Random Search:', leave=False):
        function = NP_Heap(Randomize=True, max_depth=max_depth,
                           const_prob=const_prob,
                           C_range=C_range)
        MSE_i = function.get_MSE(data)
        if MSE_i < best_function_err:
            best_solution = function
            best_function_err = MSE_i
            MSE_log.append([i, MSE_i])
        else:
            MSE_log.append([i, best_function_err])

    return best_solution, np.array(MSE_log)


###     hill climber         ####
# TODO Improve mutation this is very slow
# TODO implement operator mutations
def HC(target_data, step_search_size=128, max_depth=3, mutate_prcnt_change=.01,
       const_prob=.5, C_range=(-10, 10), given_function=None, Optimized_random=0):
    '''
    This function will search random children and move in the best direction from an optimized random start. 

    '''
    if not given_function:
        # initialize return functions
        if Optimized_random>0:  # does a quick random search to eliminate the trash
            # TODO diversity issue this should be deliberately implemented at the random function level
            Best_Function, _ = Random_Search(evaluations=Optimized_random,
                                             data=target_data,
                                             max_depth=max_depth,
                                             C_range=C_range)
        else:
            Best_Function = NP_Heap(length=32)
            Best_Function.Random_Heap(max_depth=max_depth,
                                      const_prob=const_prob,
                                      C_range=C_range)
    else:
        Best_Function = given_function

    Min_MSE = Best_Function.get_MSE(target_data)
    MSE_log = []
    MSE_log.append([Optimized_random if Optimized_random>0 else 1, Min_MSE])
    Improved = True
    step_num = 0
    while Improved:
        Improved = False  # to be flagged true if any of the children is better than the parent
        # parent of all children to search steps based on the curent best
        gen_parent = Best_Function.copy()
        for _ in tqdm(range(step_search_size), desc='Stepping:', leave=False):
            # loops N times testing nearby points
            step = gen_parent.copy()
            step.Constant_Mutation(change_prcnt=mutate_prcnt_change)
            step_MSE = step.get_MSE(target_data)
            if step_MSE < Min_MSE:
                # this will track best step at a position.
                Best_Function = step
                Min_MSE = step_MSE
                Improved = True  # only needs to happen once to be overritedn
        step_num += 1
        MSE_log.append([step_num*step_search_size, Min_MSE])

        #print('loop ', step_num, ' DONE')
        #print('Best child \n',Best_Function)
        #print('min MSE', Min_MSE)

    return Best_Function, MSE_log
# Random starts


def RSHC(Starts, target_data, step_search_size=128, max_depth=3, mutate_prcnt_change=.02,
         const_prob=.5, C_range=(-10, 10), Optimized_random=0):

    improvement_log = []
    total_evals = 0
    Best_MSE = 1e7
    best_function = None
    for i in tqdm(range(Starts), desc='RSHC:'):
        function, mse_arr = HC(step_search_size=step_search_size,
                               target_data=target_data,
                               mutate_prcnt_change=mutate_prcnt_change,
                               max_depth=max_depth,
                               const_prob=const_prob,
                               C_range=C_range,
                               Optimized_random=Optimized_random)
        evals, best_MSE_i = mse_arr[-1]
        total_evals += evals
        print(total_evals)
        if best_MSE_i < Best_MSE:
            improvement_log.append([total_evals, best_MSE_i])
            best_function = function
            Best_MSE = best_MSE_i
    return best_function, improvement_log


# Pickle & save data somewhere
def save_run(Population, data_name, folder="saved_runs", optional=''):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get the current date and time
    current_date = datetime.datetime.now()
    # Format the date without the year
    date_str = current_date.strftime("%b-%d_%H-%M")

    # Define the file name
    file_name = f"{folder}/{data_name}_date_{date_str}_{optional}.pkl"

    # Serialize and save the object to the file
    with open(file_name, 'wb') as file:
        pickle.dump(Population, file)


if __name__ == '__main__':
    # PARALLEL HC LEARNING CURVE PLOT
    level = input("Enter level (Bronze.txt, Silver.txt, Gold.txt): ")
    data = np.loadtxt(level, dtype=float, delimiter=',')
    Y_range = (np.min(data[:, 1]), np.max(data[:, 1]))
    iterations = 5
    evals = input("Enter number of starts (100): ")
    evals = int(evals)

    # Runs
    Population = np.full(iterations, None, dtype=object)
    for i in tqdm(range(iterations), desc='Iterations:', leave=False):
        best_function, performance_log = RSHC(Starts=evals,
                                            step_search_size=100,
                                            mutate_prcnt_change=.08,
                                            target_data=data,
                                            max_1100depth=5,
                                            C_range=Y_range,
                                            Optimized_random=100)
        Population[i] = ((best_function, performance_log))

    # Save runs
    save_run(Population, 'HC', folder='Results_{}'.format(level), optional='{}_tests_{}_evals'.format(iterations, evals))