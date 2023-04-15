from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd



#loading the dataset
bc = load_breast_cancer()
print(bc)
x = scale(bc.data)
print(x) 
y = bc.target
print(y)

#training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=6)
model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)
predictions = model.predict(x_test)
labels = model.labels_
accuracy = accuracy_score(y_test, predictions)

print('labels:', labels)
print('predictions:', predictions)
print('accuracy:',accuracy)
print('actual:',y_test)
print(pd.crosstab(y_train, labels))
