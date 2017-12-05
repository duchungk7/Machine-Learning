# Data Preprocessing Template

"""Step01: download data"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Step03: Encoding data 
# Encoding categorical
# transfer to number
# France  0
# Spain   1
# Germany 2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])


# transfer to one hot
# France  1   0   0
# Spain   0   1   0
# Germany 0   0   1
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dumy Variable Trap
X = X[:, 1:]

#Step04: Split Training set and Test set
# Splitting the dataset into the Training set and Test set
# test_size - 0.2
# Train set  80%
# Test set 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set values
y_pred = regressor.predict(X_test)

"""
# 反向淘汰
# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X_train = np.append(arr = np.ones((40, 1)).astype(int), values = X_train, axis = 1)
X_opt = X_train [:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
"""



# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.savefig('training_set.png', dpi=100)
plt.show()
#
"""
#Step02: Missing data processing 
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "median", axis = 0)
imputer= imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Step03: Encoding data 
# Encoding categorical
# transfer to number
# France  0
# Spain   1
# Germany 2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# transfer to one hot
# France  1   0   0
# Spain   0   1   0
# Germany 0   0   1
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# encoding y data
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#Step04: Split Training set and Test set
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Step05: Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

