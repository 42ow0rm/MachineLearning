# Apriori

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
dataset_path = "path/to/the/dataset.csv"

# Import the dataset
dataset = pd.read_csv(dataset_path, header = None)
transactions = []
for i in range(0, 7501): #upper bound excluded, lower bound included
  transaction.append([str(dataset[i,j]) for j in range(0, 20)])
  
# Training Apriori on the dataset
from apryori impor apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
""" 
  min_support = 3*7/7500    items that are rarely buyed
  min_confidence = ((80%)/2)/2
"""

# Visualising the results
results = list(rules) #sote the rules that founded by the alg in a var
