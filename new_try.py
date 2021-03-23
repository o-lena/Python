def umwandlung
import numpy as np
import ast
var = {"2017,27000,20,61.4,2.1", "2016,6200,555,28,5.5", "2016,16000,325,30.4,4"}
li = list(var)
lis = [s.split(',') for s in li]
for n in lis: 
    x = []
    x.append([ast.literal_eval(i) for i in n])
    print(x)