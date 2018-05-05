# Thompson Sampling

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Settings
dataset_path = "path/to/the/dataset.csv"
N = 10000
d = 10

# Import the dataset
dataset = pd.read_csv(dataset_path, header = None)

def histogram(items, title = 'Title', xlabel = 'x', ylabel = 'y'):
  # Visualising the results - Histogram
  plt.hist(items)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

# Implementing the Thopson Sampling
numbers_of_rewards_1 = [0] *d
numbers_of_rewards_0 = [0] *d
items_selected = []
total_reward = 0
for n in range(0, N):
  item = 0
  max_random = 0
  for i in range(0, d):
    random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, number_of_reward_0[i] + 1) 
    if random_beta > max_random:
      max_random = upper_bound
      item = i
  items_selected.append(item)
  reward = dataset.values[n, item]
  if reward ==1:
    number_of_rewards_1[item] = number_of_rewards_1[item] + 1
  else:
    number_of_rewards_0[item] = number_of_rewards_0[item] + 1
  total_reward = total_reward + reward
histogram(items_seleted, 'Histogram of ads selections', 'Items', 'Number of times each item was selected')
