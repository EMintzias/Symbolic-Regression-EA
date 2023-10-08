#HW_2 Sandbox
#%%
import numpy as np


#%%

class Function_node:
    def __init__(self, function_name, children=None, value=None):
        self.function_name = function_name
        self.children = children if children else []
        self.value = value

    def evaluate(self, variables):
        if self.value is not None:
            return self.value
        if self.function_name == 'add':
            return self.children[0].evaluate(variables) + self.children[1].evaluate(variables)
        elif self.function_name == 'mult':
            return self.children[0].evaluate(variables) * self.children[1].evaluate(variables)
        elif self.function_name == 'div':
            return self.children[0].evaluate(variables) * self.children[1].evaluate(variables)
        elif self.function_name == 'cos':
            return np.cos(self.children[0].evaluate(variables))
        elif self.function_name == 'sin':
            return np.sin(self.children[0].evaluate(variables))
        elif self.function_name == 'var':
            return variables[self.value]
        elif self.function_name == 'const':
            return variables[self.value]
        

#%%
import numpy as np

x = np.empty(5)
print(x)
x.fill(None)
x[2] = 3
print(x)
for val in x: 
    if not np.isnan(val): 
        print('non nan')
    else:
        print('nan')


if np.nan: 
    print('nan is true')



