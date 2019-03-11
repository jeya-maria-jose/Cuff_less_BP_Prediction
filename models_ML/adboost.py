import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
from sklearn.metrics import mean_squared_error, r2_score
import csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


dataset = pd.read_csv('dataset_with_physio_6_3.csv',names = ['ind','alpha','PIR', 'ptt', 'bpmax' ,'bpmin', 'hrfinal', 'ih', 'il', 'meu', 'j', 'k','l','m','n','o','p','q','r','s','p1','p2','p3','p4','p5','p6','p7','p8','p9'])


X = dataset[[  'ptt','p1','p2','p3','p4','p5','p6','p7','p8','p9']]

y = dataset[['bpmin']]

sbp = list()
dbp = list()
real_BP = list()
with open('dataset_with_physio_6_3.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter = ',')
	print csv_reader
	for row in csv_reader:
		#ptt.append(float(row[2]))
		sbp.append(float(row[4]))
		dbp.append(float(row[5]))

	
	real_BP = list()
	for i in range(len(sbp)):
		BP_actual = (2*dbp[i] + sbp[i])/3
		real_BP.append(BP_actual)
		


X_train, X_test, y_train, y_test = train_test_split(X, real_BP, test_size=0.1, random_state=1) 

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(X_train)
x_test=sc_X.transform(X_test)

rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
y_pred = regr_1.predict(X_test)
y_pre = regr_2.predict(X_test)

#print('Coefficients: \n', regressor.coef_)
# The mean squared error
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pre))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pre))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pre)))



# # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
