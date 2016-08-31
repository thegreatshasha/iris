# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:24:05 2016

@author: pawan
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import csv 
from sklearn import tree
from sklearn.datasets import load_iris
import os
from sklearn.externals.six import StringIO


iris  =load_iris() 
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)

with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
    
#os.unlink('iris.dot')

#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("iris.pdf") '