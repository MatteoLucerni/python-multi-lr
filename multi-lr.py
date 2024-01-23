import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

boston = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", sep=r'\s+', names=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PRATIO","B","LSTAT","MEDV"])

boston.head()

# correlation heatmap
sns.heatmap(boston.corr(), xticklabels=boston.columns, yticklabels=boston.columns)
plt.show()

# main columns heatmap
cols = ['RM', 'LSTAT', 'PRATIO', 'INDUS', 'MEDV']
sns.heatmap(boston[cols].corr(), xticklabels=boston[cols].columns, yticklabels=boston[cols].columns, annot=True, annot_kws={'size':10})
plt.show()

# couples' graph
sns.pairplot(boston[cols])
plt.show()

# only two colums
X = boston[['RM', 'LSTAT']].values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

print('Score: ' + str(r2_score(Y_test, Y_pred)))
print('Error: ' + str(mean_squared_error(Y_test, Y_pred)))

# all columns

X = boston.drop('MEDV', axis=1).values
Y = boston['MEDV'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# scaling data

ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

lr = LinearRegression()
lr.fit(X_train_std, Y_train)
Y_pred = lr.predict(X_test_std)

print('------- all columns ----------')
print('Score: ' + str(r2_score(Y_test, Y_pred)))
print('Error: ' + str(mean_squared_error(Y_test, Y_pred)))

print(list(zip(boston.columns, lr.coef_)))