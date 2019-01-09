import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error, r2_score


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(40, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


dataset = pd.read_csv('Total.csv',names = ['alpha','PIR', 'ptt', 'bpmax' ,'bpmin', 'hrfinal', 'ih', 'il', 'meu'])

X = dataset[['alpha','PIR', 'ptt','hrfinal', 'ih', 'il', 'meu']]

y = dataset[['bpmin']]
#X2 = dataset2[['Ptt','ih','il','hr','v','w']]
#test = datset2[['firstaxis_1','firstaxis_2','firstaxis_3','secondaxis_1','secondaxis_2','secondaxis_3','thirdaxis_1','thirdaxis_2','thirdaxis_3','eigen1','eigen2','eigen3','firstaxis_len','secondaxis_len','thirdaxis_len','c1','c2','c3','mer_ecc','eq_ecc','age']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
#print X_test

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(X_train)
x_test=sc_X.transform(X_test)


estimator = KerasRegressor(build_fn=baseline_model, epochs=1000, batch_size=10, verbose=2)


estimator.fit(x_train, y_train)
y_pred = estimator.predict(x_test)
#print y_test
#print regressor.coef_
# print('Coefficients: \n', regressor.coef_)
# # The mean squared error
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

