import pandas as pd
import json
import hjson
import numpy as np
from itertools import chain
import pdb


filename = "all_variables.json"

with open(filename, "r") as f:
    variables = json.load(f)

varQ1 = variables["Q1"]
varQ3 = variables["Q3"]
varQ4 = variables["Q4"]
varQ5 = variables["Q5"]
varQ6 = variables["Q6"]


def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [*value , *dict_1[key]]
   return dict_3

dictQ1Q3 = mergeDictionary(varQ3, varQ1)
# dictQ1Q3Q4 = mergeDictionary(dictQ1Q3, varQ4)
# dictQ1Q3Q4Q5 = mergeDictionary(dictQ1Q3Q4, varQ5)
# all = mergeDictionary(dictQ1Q3Q4Q5, varQ6)

all_items = varQ6.values()
items = list(chain.from_iterable(all_items))
print(len(items))

nrows = 22
ncols = 13

matrix = np.zeros([nrows,ncols]).astype("str")


counter = 0
for row in range(nrows):
    for col in range(ncols):
        matrix[row, col] = items[counter]
        counter +=1
        if counter > len(items)-1:
            break


df = pd.DataFrame(matrix, columns=["" for i in range(ncols)])
print(df.to_latex(index=False))
