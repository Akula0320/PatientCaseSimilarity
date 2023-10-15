import pandas as pd
from sklearn.model_selection import train_test_split
# , KFold, cross_val_score
# import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
import numpy as np
import pickle

data = pd.read_csv('diabetes.csv')

cols = ['Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity']

for col in cols:
  data[col] = (data[col].str.strip() == "Yes").astype(int)


data['Gender'] = (data['Gender'].str.strip() == "Male").astype(int)
data['class'] = (data['class'].str.strip() == "Positive").astype(int)

X = data.drop(columns = 'class')
y = data['class']
no_folds = 6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 6)
Model  = RandomForestClassifier(n_estimators = 50)
Model.fit(X_train, y_train)
pickle.dump(Model, open("Model.pkl", "wb"))
