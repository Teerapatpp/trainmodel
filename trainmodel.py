from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

iris['data']
iris['target']


X, y = iris['data'], iris['target']
dataset = np.hstack((X, y.reshape(-1,1)))

np.random.shuffle(dataset)
X_train, X_test, y_train, y_test = train_test_split(dataset[:,:4],
                                                   dataset[:,4],
                                                   test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#print("predied is : "y_pred)

#acc =accuracy_score(y_test, y_pred)
#print("Accuracy sore is :",acc )

#result = model.predict([[5.1, 3.5, 1.4, 0.2]])
#print(result)

joblib.dump(model,'iris.model')