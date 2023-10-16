# HW_2 Sandbox
# %% FUNCTIONS
from tqdm import tqdm
import numpy as np
import heapq as h
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pdb
import math
import math as m
from icecream import ic
import pandas as pd
import random
import time
# %% --------------------------------------------------
# LOAD DATA:
Bronze_data = np.loadtxt("Bronze.txt", dtype=float, delimiter=',')
Silver_data = np.loadtxt("Silver.txt", dtype=float, delimiter=',')
Gold_data = np.loadtxt("Gold.txt", dtype=float, delimiter=',')
Platinum_data = np.loadtxt("Platinum.txt", dtype=float, delimiter=',')

print('Bronze_data Range \t=', np.min(
    Bronze_data[:, 1]), '<->', np.max(Bronze_data[:, 1]))
print('Silver_data Range \t=', np.min(
    Silver_data[:, 1]), '<->', np.max(Silver_data[:, 1]))
print('Gold_data Range \t=', np.min(Gold_data[:, 1]), np.max(Gold_data[:, 1]))
print('Platinum_data Range \t=', np.min(
    Platinum_data[:, 1]), '<->', np.max(Platinum_data[:, 1]))

Y_range_Cu = (np.min(Bronze_data[:, 1]), np.max(Bronze_data[:, 1]))
Y_range_Ag = (np.min(Silver_data[:, 1]), np.max(Silver_data[:, 1]))
Y_range_Au = (np.min(Gold_data[:, 1]), np.max(Gold_data[:, 1]))
Y_range_Pt = (np.min(Platinum_data[:, 1]), np.max(Platinum_data[:, 1]))

print(len(Gold_data))
# %% -------------------------------------------------


class Function_node(object):
    # Globals to be inherited
    # last is the header indicator (indexing from 1!)
    node_types = ('+', '-', '*', '/', '^', 'sin', 'cos',
                  'const', 'x', 'X', 'var', 'ERR')
    dual_operators = ('+', '-', '*', '/', '^')
    single_operators = ('sin', 'cos')

    # TODO

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

        '''
        if self.function_name == 'const':
            if value == None:
                self.value = 0  # TODO: np.random.uniform(yrange)
            else:
                self.value = value
        '''

        if self.function_name in self.dual_operators:
            self.req_num_childs = 2
        elif self.function_name in self.single_operators:
            self.req_num_childs = 1
        else:
            self.req_num_childs = 0

    # METHODS:
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

    def __str__(self):  # print statement
        if self.function_name == 'const':
            return f"{round(self.value,2)}"
        elif self.function_name in ['var', 'x', 'X']:
            if self.value:
                return f"x={self.value}"
            else:
                return "X"
        else:
            return f"{self.function_name}"


# %% ----------------------------------------------------------------
# ***  NP_HEAP CLASS ***

# inspiration: https://github.com/swacad/numpy-heap/blob/master/heap.py
# Starting at index 1~
class NP_Heap(Function_node):
    def __init__(self, length=2, Heap=None, Randomize=False, max_depth=4, const_prob=.45, C_range=(-10, 10)):
        self.heap = np.full(length, None, dtype=object)
        # ALL Operators ('+', '-', '*', '/', '^', 'sin', 'cos')
        self.operators = ('*', '+', '-')
        self.trig_operators = ('sin', 'cos')
        self.non_operator = ('x', 'const')

        # Creating ARR from given heap list
        '''
        if type(Heap) == None: 
            
        else: 
            self.heap = np.full(len(Heap), None, dtype=object)
            for i,val in enumerate(Heap): #heap is a list of values indexed at 0
                self.heap[i+1] = val 
        '''

        if Randomize:
            self.Random_Heap(max_depth=max_depth,
                             const_prob=const_prob,
                             C_range=C_range)

    def __str__(self):
        return self.print_heap_tree()

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

    # Why not just do this?
    # def has_root(self):
    #   return self.heap[0] != None

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

    def show_function(self):  # TODO
        # evaluates a node given its index
        heap_str = [str(node) for node in self.heap]
        depth = None  # some function of length (1/2)
        function_str = None
        return None

    def plot_approximation(self, X_range=(0, 10), target_data=None, y_pred=None, ):
        X_arr = np.linspace(X_range[0], X_range[1], 1000)

        if type(target_data) != None:
            plt.plot(target_data[:, 0], target_data[:, 1])

        if not y_pred:
            y_pred = [self.evaluate(X=x) for x in X_arr]

        plt.plot(X_arr, y_pred, 'b')
        plt.show()
        return None
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
            # TODO heap depth
            if self.depth() >= max_depth - 1 or np.random.rand() < const_prob:  # no operators terminate with constants
                node_name = np.random.choice(['x', 'const'])
                node_val = np.random.uniform(
                    C_range[0], C_range[1]) if node_name == 'const' else None
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
        if R_child.function_name in self.operators:
            self.Random_Heap(2*Index+1, max_depth=max_depth,
                             const_prob=const_prob, C_range=C_range)

    # ****  EVALUATE A NODE ***:
    def evaluate(self, node_ind=1, X=None):  # tree root = 1
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

        elif L_child.function_name in ['var', 'x', 'X']:
            L_child.value = X

        if type(R_child) is None:
            pass

        elif R_child.function_name in self.operators:
            R_child.value = self.evaluate(node_ind=2*node_ind+1, X=X)

        elif R_child.function_name in ['var', 'x', 'X']:
            R_child.value = X

        # terminating state: both children are constandts (floats) or Nan (with at least a constant) after being evaluated
        node_operator = self.heap[node_ind].function_name
        try:
            if Function_node not in children_types:
                raise TypeError(
                    f"Invalid children type for operator: {node_operator} \n\t L/R children are: {(L_child, R_child)}")

            # i.e its sin, cos, tan etc (as defined above
            elif node_operator in self.trig_operators:

                if None not in children_types:
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
        msg_out = (
            f"at the root evaluating with:\n\t - parent node index {node_ind}"
            f"\n\t - operator {node_operator}"
            f"\n\t - with children {[str(x) for x in self.heap[[L_indx, R_indx]]]}"
            f"\n result passed {node_val}"
        )
        # print(msg_out)
        return node_val
    # **** END NODE EVAL   ***

    def MSE(self, point_cloud, plotting=False):
        # RECALL: MSE = (1/n) * Σ(actual – forecast) ^2
        # TODO subsection (skip every other point)
        X_arr = point_cloud[:, 0]
        y = point_cloud[:, 1]
        y_pred = np.array([self.evaluate(X=x) for x in X_arr])
        MSE = np.sum(np.square(y_pred-y)/y.shape[0])

        if plotting:
            self.plot_approximation(X_arr, y_pred, y_true=y)
        return MSE

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
    def Constant_Mutation(self, change_prcnt=.02, min_mutation=.05):
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

    def NUCLEAR_DIVERSIY(self):
        #
        pass

    def subtree(self, indx):
        subtree_ind = [indx]
        subtree_depth = self.depth() - int(m.floor(m.log2(indx)))  # heap depth - node depth
        for i in range(1, subtree_depth+1):
            for j in range(indx*2**i, (indx+1)*2**i):
                subtree_ind.append(j)

        return subtree_ind, self.heap[subtree_ind]
# %%
# Testing new heap functionality


# %% RANDOM SEARCH
###### RANDOM & HC ###############


def Random_Search(evaluations, data, max_depth=3, const_prob=.5, C_range=(-10, 10)):
    MSE_log = []
    best_solution = None

    best_function_err = 1e9
    for i in range(evaluations):
        function = NP_Heap(Randomize=True, max_depth=max_depth,
                           const_prob=const_prob,
                           C_range=C_range)
        MSE_i = function.MSE(data)
        if MSE_i < best_function_err:
            best_solution = function
            best_function_err = MSE_i
            MSE_log.append([i, MSE_i])

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
        if Optimized_random:  # does a quick random search to eliminate the trash
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

    Min_MSE = Best_Function.MSE(target_data)
    MSE_log = []
    Improved = True
    step_num = 0
    while Improved:
        Improved = False  # to be flagged true if any of the children is better than the parent
        # parent of all children to search steps based on the curent best
        gen_parent = Best_Function.copy()
        for _ in range(step_search_size):
            # loops N times testing nearby points
            step = gen_parent.copy()
            step.Constant_Mutation(change_prcnt=mutate_prcnt_change)
            step_MSE = step.MSE(target_data)
            if step_MSE < Min_MSE:
                # this will track best step at a position.
                Best_Function = step
                Min_MSE = step_MSE
                Improved = True  # only needs to happen once to be overritedn
        step_num += 1
        MSE_log.append([step_num*step_search_size, Min_MSE])

        print('loop ', step_num, ' DONE')
        #print('Best child \n',Best_Function)
        print('min MSE', Min_MSE)

    return Best_Function, MSE_log
# Random starts


def RSHC(Starts, target_data, step_search_size=128, max_depth=3, mutate_prcnt_change=.02,
         const_prob=.5, C_range=(-10, 10), Optimized_random=0):

    improvement_log = []
    total_evals = 0
    Best_MSE = 1e7
    best_function = None
    for i in range(Starts):
        print('**** START = ', i, ' *****')
        function, mse_arr = HC(step_search_size=step_search_size,
                               target_data=target_data,
                               mutate_prcnt_change=mutate_prcnt_change,
                               max_depth=max_depth,
                               const_prob=const_prob,
                               C_range=C_range,
                               Optimized_random=Optimized_random)
        evals, best_MSE_i = mse_arr[-1]
        total_evals += evals
        if best_MSE_i < Best_MSE:
            improvement_log.append([total_evals, best_MSE_i])
            best_function = function
            Best_MSE = best_MSE_i
    return best_function, improvement_log
# %%

#  RUN RANDOM SEARCH


start = time.time()
function, mse_arr = Random_Search(2000,
                                  data=Bronze_data,
                                  max_depth=3,
                                  C_range=Y_range_Cu)
runtime = time.time() - start
function.plot_approximation(target_data=Bronze_data)


# %% RUN RSHC
best_function, performance_log = RSHC(Starts=10,
                                      step_search_size=25,
                                      mutate_prcnt_change=.08,
                                      target_data=Bronze_data,
                                      max_depth=3,
                                      C_range=Y_range_Cu,
                                      Optimized_random=100)
print(best_function, '\n MSE= ', best_function.MSE(Bronze_data))
best_function.plot_approximation(target_data=Bronze_data)


# %%
# PRINT RSHC RESULTS
print(best_function, '\n MSE= ', best_function.MSE(Bronze_data))
best_function.plot_approximation(target_data=Bronze_data)

# %%
## **** EVOLUTIONARY ALGOS ***** ##
# TODO test turnament selection
# TODO robust crossover (not stress tested)
# TODO implement better mutation (currently just constants by +_2% (very slow))


def initialize_popuplation(pop_size, tree_depth=4, const_prob=.35, init_Constant_Range=(-10, 10)):
    # returns population of functions in
    if pop_size % 2:
        pop_size += 1  # ensure population num is even

    Population = np.full(pop_size, None)
    for i in range(pop_size):
        Population[i] = NP_Heap(Randomize=True,
                                max_depth=tree_depth,
                                const_prob=const_prob,
                                C_range=init_Constant_Range)
    return Population


def selection(population, data):
    # returns parent-children pairs
    # Simple turnament selection:
    def Ranked(population, data, min_percentile=0, max_percentile=50):
        fitness_ind = np.argsort(np.array([F.MSE(data) for F in population]))

    def turnament_selection(population, data, bracket_size=8):
        # returs best in subset (probability that top x% is in subset(s) is obv 1-(1-x)^s )
        Subset = np.random.choice(population, bracket_size)
        fitness = np.array([F.MSE(data) for F in Subset])
        return Subset[np.argmin(fitness)]

    def percentile_selection(population, data, subset_size=10, percentile=40):
        # returs Nth best percentile in the subset #TODO automate subset size best on percentail to a 95% CI
        Subset = np.random.choice(population, subset_size)
        fitness_ind = np.argsort(np.array([F.MSE(data) for F in Subset]))
        print(f'fitness 2 : \n {fitness_ind}')
        return Subset[fitness_ind[round(subset_size*(1-percentile)/100)]]

    # basic strategy: turnament best
    # TODO cross with shitty peple
    Parent_1 = turnament_selection(population, data)
    Parent_2 = turnament_selection(population, data)

    return Parent_1, Parent_2,


def Crossover(Parent_1, Parent_2, at_node=None):
    P1, P2 = Parent_1, Parent_2
    # confirm both have parent or choose form a both parent
    if at_node == None:
        p1_populated_ind = np.arange(P1.heap.size)[P1.heap != None]
        p2_populated_ind = np.arange(P2.heap.size)[P2.heap != None]
        common_ind = np.intersect1d(p1_populated_ind, p2_populated_ind)
        # TODO ELIMINATE ROOT SWAPS
        at_node = np.random.choice(common_ind[common_ind != 1])

    p1_subtree_ind, p1_subtree_nodes = P1.subtree(at_node)
    p2_subtree_ind, p2_subtree_nodes = P2.subtree(at_node)

    # TODO add conditionals for growing array in case there is an indexing issue.
    Len = max(P1.heap.size, P2.heap.size)
    C1 = P1.copy(Len)
    C2 = P2.copy(Len)

    C1.heap[p2_subtree_ind] = p2_subtree_nodes
    C2.heap[p1_subtree_ind] = p1_subtree_nodes

    return C1, C2


def Population_Crossover(Population, target_data):
    p_size = len(Population)
    new_pop = np.full(p_size, None, dtype=object)

    for i in tqdm(range(0, p_size, 2), desc=f'Population Crossover (size = {p_size})'):
        Parent_1, Parent_2 = selection(population=Population, data=target_data)
        Child_1, Child_2 = Crossover(Parent_1, Parent_2, at_node=None)
        new_pop[i], new_pop[i+1] = Child_1, Child_2
    return new_pop

# TODO TEST


def Population_Mutation(Population,
                        Constant_Mutation=True, change_prcnt=.02, min_mutation=.1,
                        Operator_Mutation=True, swap_num=1):
    for function in tqdm(Population, desc='Population Mutation'):
        if Constant_Mutation:
            function.Constant_Mutation(
                change_prcnt=change_prcnt, min_mutation=min_mutation)
        if Operator_Mutation:
            function.Operator_Mutation(swap_num=swap_num)

    return Population


def EP_Symbolic_Rgresion(target_data, pop_size=10, generations=50):
    Population = initialize_popuplation(pop_size=pop_size)
    improvement_log = []
    population_MSEs = []
    num_evaluations = 0
    best_heap = None
    for i in range(generations):
        print(f'Generation {i+1}')
        # Crossover does both selection and crossover for a new population
        Population = Population_Crossover(Population, target_data)
        Population = Population_Mutation(Population)

        fitness_arr = np.array([F.MSE(target_data) for F in Population])
        pop_fitness_ind = np.argsort(fitness_arr)  # min is best MSE

        # TODO embed this info into the selection.
        num_evaluations += pop_size
        evaluated_populations = fitness_arr[pop_fitness_ind]
        best_function = Population[pop_fitness_ind[0]]
        print(f'Best MSE this gen =', best_function.MSE(target_data))
        improvement_log.append(
            [num_evaluations, best_function, best_function.MSE(target_data)])
        population_MSEs.append(evaluated_populations)
        # WHAT TO RETURN FOR PLOTTIONG:

    Progress_array = np.array(improvement_log)
    Column_labels = ['Evaluations', 'Best agent', 'Lowest MSE']
    Progress_df = pd.DataFrame(Progress_array, columns=Column_labels)
    Progress_df["Pop_MSEs"] = population_MSEs
    last_Population = Population  # last population in order

    return Progress_df, last_Population_ranked


# %%
pop_test = initialize_popuplation(pop_size=3)
for f in pop_test:
    print(f)
pop_test = Population_Mutation(pop_test)
for f in pop_test:
    print(f)


# %%
# TEst GP:
Progress_df, final_pop_ordered = EP_Symbolic_Rgresion(
    Bronze_data, pop_size=200, generations=50)

# %%
Progress_df.head()

best_function = Progress_df['Best agent'].iloc[-1]
print(best_function)
best_function.plot_approximation(target_data=Bronze_data)
# %%
# Testing EP
np.random.seed(25)
h1 = NP_Heap(Randomize=True, max_depth=4, const_prob=.3)
h2 = NP_Heap(Randomize=True, max_depth=4, const_prob=.3)
h2.Constant_Mutation(change_prcnt=.02, min_mutation=.05)
# print(h2)
# h1.print_arr()
# h2.print_arr()

s1_ind, s1_vals = h1.subtree(2)
s2_ind, s2_vals = h2.subtree(2)

C1, C2 = Crossover(h1, h2, at_node=2)

print('before')
print(h1)
print(h2)
print('After')
print(C1)
print(C2)

'''
#testing the subtree print
print(s_ind)
tree_str = [None]
for v in s_vals: 
    tree_str.append(v)
    
new_h = NP_Heap(length=len(tree_str))
new_h.heap = tree_str
print(new_h)
'''

# %%

# Define the number of iterations in your loop
total_iterations = 100

# Create a loop with a progress bar
for i in tqdm(range(total_iterations), desc="Processing"):
    function, mse_arr = Random_Search(10,
                                      data=Bronze_data,
                                      max_depth=3,
                                      C_range=Y_range_Cu)

    pass
