import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#pd.read_csv for csv docs
#import test data and assign independant variables and the dependant variable
dataset = pd.read_excel('Data.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(x)
#print(y)

#take care of missing data, no value aka 'nan' and input the mean of the coloumn
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#choose coloumns to identify missing values and transform (input mean)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#encode independant variables, turn string values to binary
#[0] being the coloumn index we went to encode
#passthrough = include the other coloums
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#print(x_test)

#encode dependant variable in this example turnning yes/no to 1 or 0
le = LabelEncoder()
y = le.fit_transform(y)

#from the data, split into test data and training data. Test data only needs to be 20% as we want the bulk of the data for training and create an accurate model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)