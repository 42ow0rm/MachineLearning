# Upper confidence bound UCB

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
impoert math

# Settings
dataset_path = "path/to/the/dataset.csv"
N = 10000
d = 10

# Import the dataset
dataset = pd.read_csv(dataset_path, header = None)

# Implementing Random Selection
def RandomSelection():
    items_selected = []
    total_reward = 0
    for i in range(0,N):
      item = random.randrange(10)
      items_selected.append(item)
      reward = dataset.values[n, item]
      total_reward = total_reward + reward
      histogram(items_seleted, 'Histogram of ads selections', 'Items', 'Number of times each item was selected')
      
def histogram(items, title = 'Title', xlabel = 'x', ylabel = 'y'):
    # Visualising the results - Histogram
    plt.hist(items)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Implementing the UCB
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
items_selected = []
total_reward = 0
for n in range(0, N):
  item = 0
  max_upper_bound = 0
  for i in range(0, d):
    if (numbers_of_selection[i] > 0):
      average_reward = sums_of rewards[i] / numbers_of_selections[i]
      delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selection[i])
      upper_bound = average_reward + delta_i
    else:
      upper_bound = 1e400
    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      item = i
  items_selected.append(item)
  numbers_of_selections[item] = numbers_of_selections[item] + 1
  reward = dataset.values[n, item]
  sums_of_rewards[items] = sums_of_rewards[items] + reward
  total_reward = total_reward + reward
histogram(items_seleted, 'Histogram of ads selections', 'Items', 'Number of times each item was selected')
  
