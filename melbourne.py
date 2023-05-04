import pandas as pd
import numpy as np
from sklearn. model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt



datasets = pd.read_csv('insurance.csv')
#print(datasets.head())
#sex
label = LabelEncoder()
label.fit(datasets.sex.drop_duplicates())
datasets.sex = label.transform(datasets.sex)

#smoker
label.fit(datasets.smoker.drop_duplicates())
datasets.smoker = label.transform(datasets.smoker)
#training the mmodel
label.fit(datasets.region.drop_duplicates())
datasets.region = label.transform(datasets.region)

print(datasets.head())

#creating our mode
x_lin = datasets.drop(['charges'], axis=1)
y_lin = datasets[['charges']]

x_lin_train, x_lin_test, y_lin_train, y_lin_test = train_test_split(x_lin, y_lin, test_size=0.2, random_state=42)
linear_model = LinearRegression()

linear_model.fit(x_lin, y_lin)
for idx, col_name in enumerate(x_lin_train.columns):
    print('the coefficient for {} is {}'.format(col_name, linear_model.coef_[0][idx]))
