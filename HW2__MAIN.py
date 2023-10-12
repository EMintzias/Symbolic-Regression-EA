# HW_2 Sandbox
# %%
import numpy as np
import heapq as h
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pdb
import math
import random
# %% --------------------------------------------------
# LOAD DATA:
Bronze_data = np.loadtxt("Bronze.txt", dtype=float, delimiter=',')
Silver_data = np.loadtxt("Silver.txt", dtype=float, delimiter=',')
Gold_data = np.loadtxt("Gold.txt", dtype=float, delimiter=',')
Platinum_data = np.loadtxt("Platinum.txt", dtype=float, delimiter=',')

print('Bronze_data Range =', np.min(
    Bronze_data[:, 1]), np.max(Bronze_data[:, 1]))
print('Silver_data Range =', np.min(
    Silver_data[:, 1]), np.max(Silver_data[:, 1]))
print('Gold_data Range =', np.min(Gold_data[:, 1]), np.max(Gold_data[:, 1]))
print('Platinum_data Range =', np.min(
    Platinum_data[:, 1]), np.max(Platinum_data[:, 1]))

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

    def __str__(self):  # print statement
        if self.function_name == 'const':
            return f"{self.value}"
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
    def __init__(self, length=2):
        self.heap = np.full(length, None, dtype=object)
        # ALL ('+', '-', '*', '/', '^', 'sin', 'cos')
        self.operators = ('*', '+', '-')
        self.trig_operators = ('sin', 'cos')
        self.non_operator = ('x', 'const')

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
        return 2 * parent_idx, 2 * parent_idx + 1

    def get_children(self, parent_idx):
        child_1_idx, child_2_idx = self.get_children_idx(parent_idx)
        return self.heap[child_1_idx], self.heap[child_2_idx]

    def get_children_type(self, parent_idx):
        L, R = self.get_children(parent_idx)
        return type(L), type(R)
    '''
    # INDEXING AT 0:

    def get_parent_idx(self, child_idx):
        return (child_idx-1) // 2

    def get_parent(self, child_idx):
        parent_idx = self.get_parent_idx(child_idx)
        return self.heap[parent_idx]

    def get_left_child_idx(self, parent_idx):
        return 2 * parent_idx + 1

    def get_right_child_idx(self, parent_idx):
        return 2 * parent_idx + 2

    def get_left_child(self, parent_idx):
        left_child = self.heap[self.get_left_child_idx(parent_idx)]
        return left_child

    def get_right_child(self, parent_idx):
        right_child = self.heap[self.get_right_child_idx(parent_idx)]
        return right_child

    def get_children_idx(self, parent_idx):
        return 2 * parent_idx + 1, 2 * parent_idx + 2

    def get_children(self, parent_idx):
        child_1_idx, child_2_idx = self.get_children_idx(parent_idx)
        return self.heap[child_1_idx], self.heap[child_2_idx]
    '''

    def get_depth(self):
        return int(math.floor(math.log2(len(self.heap) + 1)))  # -1

    def has_root(self):
        if self.heap[0] == None:
            return False
        else:
            return True
    # Why not just do this?
    # def has_root(self):
    #   return self.heap[0] != None

    def add_root(self, new_node):
        self.heap[0] = new_node

 # INSERT FUNCTION
    def insert(self, parent_indx, node_obj, position=None):
        # check for size availibility or resize

        # check heap length if short, doubles array until acceptable
        while self.heap.size - 1 < self.get_right_child_idx(parent_indx):
            #print('doubled',self.heap.size ,self.get_right_child_idx(parent_indx)  )
            self.heap = np.append(self.heap,
                                  np.full(self.heap.size, None).astype(Function_node))

        # TODO raise error for operators not in the predefined list of self.operators:

        L_indx, R_indx = self.get_children_idx(parent_indx)

        # if a position is provided it will insert with replacemnt
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
            print('Parent children filled')
            #self.insert(self,parent_indx = L, value = value )
        return None

  # ADD CHILD FUNCTION

    def add_child(self, parent_idx, new_node):
        #print("trying to add child for {}".format(parent_idx))
        # Check if parent can add child
        # True if can_add_child
        if self.heap[parent_idx].can_add_child():
            # Keep track of added child in node

            # Grow array if necessary:
            while self.heap.size - 1 < self.get_right_child_idx(parent_idx):
                #print('doubled',self.heap.size ,self.get_right_child_idx(parent_indx)  )
                self.heap = np.append(self.heap,
                                      np.full(self.heap.size, None).astype(Function_node))

            # Get child number [either 1 or 2 since we're adding above]
            child_num = self.heap[parent_idx].num_childs
            print('inserting at index', 2*parent_idx+child_num+1, child_num)
            self.heap[2*parent_idx+child_num+1] = new_node

            '''
            if 2*parent_idx+child_num < len(self.heap):
                self.heap[2*parent_idx+child_num] = new_node
            # Resize if needed
            else:  # TODO insert type of heap growth for optimization?
                prev_n = len(self.heap)
                prev_heap = self.heap
                # Create new heap that is 'full' to depth where new child will be
                self.heap = np.full(
                    2**(math.floor(math.log2(2*parent_idx+child_num + 1))+1)-1, None, dtype=object)
                self.heap[:prev_n] = prev_heap
                self.heap[2*parent_idx+child_num] = new_node
            '''
        # Can't add child
        else:
            # What to do here?
            #print("can't add child")
            pass

 # ---------------------------------------------

 ####### DISPLAY FUNCTIONS ##########
    def print_heap_tree(self, index=0, prefix="", is_left=True):  # CALLED BY __str__ method
        output = ""
        if index < len(self.heap) and self.heap[index] is not None:
            # Process right child first (going down)
            output += self.print_heap_tree(index=2*index+2, prefix=prefix + (
                "|   " if is_left else "    "), is_left=False)
            # Add the current node to the output
            output += prefix + "|-- " + str(self.heap[index]) + "\n"
            # Process left child
            output += self.print_heap_tree(index=2*index+1, prefix=prefix + (
                "|   " if not is_left else "    "), is_left=True)
        # else:
            # Indicate the absence of a node with [...]
            # output += prefix + "|-- [...]\n"
        return output

    def print_arr(self):  # shows the heap and indexes
        heap_str = [str(node) for node in self.heap]
        ind_arr = np.arange(self.heap.size)
        print(np.stack((ind_arr[1:], heap_str[1:])))
        return None

    def show_function(self):  # TODO
        # evaluates a node given its index
        heap_str = [str(node) for node in self.heap]
        depth = None  # some function of length (1/2)
        function_str = None
        return None

    def plot_approximation(self, X_arr, y_pred=None, y_true=None):
        if not y_pred:
            y_pred = [self.evaluate(X=x) for x in X_arr]

        plt.plot(X_arr, y_pred, 'b')

        if y_true:
            plt.scatter(X_arr, y, c='r')

        return None
 # ---------------------------------------------

 ####### Main functionalities ###########
    def randomize_heap(self, parent_idx=0, constant_prob=.6, y_range=(0, 5)):
        # Check if heap has root
        if self.has_root() == False:
            # Add operator rood node if not
            # ******* WILL MAYBE WANT TO CHANGE ********
            self.add_root(Function_node(np.random.choice(self.operators)))

        # print("...")
        #print("Depth:{}\nTree: \n{}".format(self.get_depth(), self.__str__()))

        # Kepp adding subtrees up to depth = 1
        if self.get_depth() < 1:
            # Add random nodes until sub tree is full
            while self.heap[parent_idx].can_add_child():
                rand_val = np.random.rand()
                print(rand_val, 'less than?', constant_prob)
                if rand_val < constant_prob:
                    name = random.choice(self.non_operator)
                else:
                    name = random.choice(self.operators)

                val = random.uniform(
                    y_range[0], y_range[1]) if name == 'const' else None
                new_node = Function_node(name, val)
                print('adding', name, ' to  ', parent_idx)
                self.print_arr()
                self.add_child(parent_idx, new_node)
                self.print_arr()

            # Recursively add random subtrees for all children
            # If subtree is a non_operator it will not loop as num_childs = 0
            for i in range(self.heap[parent_idx].num_childs):
                self.randomize_heap(2*parent_idx + i+1)

        # Make sure all operators required number of operands
        # Thus max depth of tree = 2
        else:
            # Add random operands for all children that require them
            while self.heap[parent_idx].can_add_child():
                typ = random.choice(self.non_operator)
                self.add_child(parent_idx, Function_node(
                    typ, value=random.uniform(0, 5) if typ == 'const' else None))
            # for i in range(self.heap[parent_idx].num_childs):
            #    typ = random.choice(self.non_operator)
            #    self.add_child(2*parent_idx + (i+1), Node(typ, value=random.uniform(0,5) if typ == 'const' else None))

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
        X_arr = point_cloud[:, 0]
        y = point_cloud[:, 1]
        y_pred = np.array([self.evaluate(X=x) for x in X_arr])
        MSE = np.sum(np.square(y_pred-y)/y.shape[0])

        if plotting:
            self.plot_approximation(X_arr, y_pred, y_true=y)
        return MSE

 # EP: MUTATION ########## :
    '''
        IDEAS: 
            - Most vanilla: search through heap for the constants and +_ by X small % of the value
            - Adding operations: for chain terminating variables, replace them with the + operator 
                                 and add a small constant to the variable
            - Mutation schedules: annealing or adaptive mutation (increases if stagnating decreases if progress)
            - Optimization by consolidating into one function so we dont have to traverse heap. (in practice heap should be small!)
    '''

    def Constant_Mutation(self, change_prcnt=.01):  # TODO
        # small change to constants and addition/substraction to variables
        # basic: for node in heap, if name = const +_ X%
        for i, node in enumerate(self.heap):
            if type(node) == Function_node and node.function_name == 'const':
                mutation_size = change_prcnt*node.value
                node.value = node.value + mutation_size * \
                    np.random.choice([-1, 1])
                self.print_arr()
        return None

    def Operator_mutation(self, number):
        for i, node in enumerate(self.heap):
            if type(node) == Function_node:  # only operate over nodes....
                if node.function_name == '*':
                    self.print_arr()
                    node.function_name = '/'
                    self.print_arr()
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


##### HEAP CLASS DONE! ####
# %%
# Testing Mutation
M1 = NP_Heap(length=2)
M1.heap[0] = Function_node('ERR')
M1.heap[1] = Function_node('+')
M1.insert(parent_indx=1, position='L', node_obj=Function_node('*'))
M1.insert(parent_indx=1, position='R',
          node_obj=Function_node('const', value=1))
M1.insert(parent_indx=2, position='L', node_obj=Function_node('x'))
M1.insert(parent_indx=2, position='R',
          node_obj=Function_node('const', value=3.14))
M1.insert(parent_indx=3, position='L', node_obj=Function_node('var'))
M1.insert(parent_indx=3, position='R',
          node_obj=Function_node('const', value=21))
M1.Constant_Mutation()


M1.print_arr()


# %% ----------------------------------
# TESTING
t1 = NP_Heap(length=2)
t1.heap[0] = Function_node('ERR')
t1.heap[1] = Function_node('+')
t1.insert(parent_indx=1, position='L', node_obj=Function_node('sin'))
t1.insert(parent_indx=1, position='R',
          node_obj=Function_node('const', value=1))
t1.insert(parent_indx=2, position='L', node_obj=Function_node('var'))
t1.insert(parent_indx=2, position='R', node_obj=Function_node('var'))
t1.insert(parent_indx=3, position='R',
          node_obj=Function_node('const', value=3.14))


t1.evaluate(X=0)
x_arr = np.linspace(0, 10, 100)

y = [t1.evaluate(X=x) for x in x_arr]
t1.heap[6] = None
t1.print_arr()
#plt.plot(x_arr, y)

#t1.plot_approximation(np.linspace(-5, 5, 100))

# %% -----------------------------------------------------------

data = Bronze_data
x = Bronze_data[:, 0]
y = Bronze_data[:, 1]
plt.plot(x, y)

y_pred = np.array([(x-3.5)**2 - 15.2 for x in Bronze_data[:, 0]])
plt.plot(x, y_pred, 'r')
MSE = np.sum(np.square(y_pred-y)/y.shape[0])
print('MSE = ', MSE)

# %%

# Create a NumPy array (for demonstration purposes)
arr = np.array([[1, 2, 3],
                [4, 55, 6],
                [7, 888, 9]])

# Get the maximum width of each column
max_width = np.max([len(str(item)) for item in arr])

# Use savetxt with custom formatting
with np.printoptions(linewidth=np.inf, formatter={'all': lambda x: f'{x:{max_width}}'}):
    np.savetxt('formatted_array.txt', arr, fmt='%d')
    print(arr)
