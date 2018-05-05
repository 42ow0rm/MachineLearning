# Natural Language Processing

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Settings
dataset_path = "path/to/the/dataset.tsv"

# Import the dataset
dataset = pd.read_csv(dataset_path, delimiter = '\t', quoting = 3)

# for the clean_text fctn
import re
import nltk #contains a list of irrelevant words
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def clean_text(text = ''):
  clean_text = re.rub('[^a-zA-Z]', ' ', text) 
  clean_text = clean_text.lower()
  clean_text.split()
  clean_text = [ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
  clean_text = ' '.join(clean_text)
  return clean_text
  
# Cleaning the texts
corpus = []
for i in range(0, 1000):
  corpus.append(clean_text(dataset['Review'][i]))
  
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CuontVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Spitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y. test_size = 0.25)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Prediction the Test set result
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
