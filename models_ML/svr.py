import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
from sklearn.metrics import mean_squared_error, r2_score

import csv
dataset = pd.read_csv('cleaned_further.csv',names = ['alpha','PIR', 'ptt', 'bpmax' ,'bpmin', 'hrfinal', 'ih', 'il', 'meu', 'j', 'k','l','m','n','o','p','q','r','s'])


X = dataset[['alpha','PIR', 'ptt']]

y = dataset[['bpmin','bpmax']]

sbp = list()
dbp = list()
real_BP = list()
with open('cleaned_further.csv', 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter = ',')
	print csv_reader
	for row in csv_reader:
		#ptt.append(float(row[2]))
		sbp.append(float(row[3]))
		dbp.append(float(row[4]))

	real_BP = list()
	for i in range(len(sbp)):
		BP_actual = (2*dbp[i] + sbp[i])/3
		real_BP.append(BP_actual)


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
#y=sc_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, real_BP, test_size=0.2, random_state=0) 

# from sklearn.preprocessing import StandardScaler
# sc_X=StandardScaler()
# x_train=sc_X.fit_transform(X_train)
# x_test=sc_X.transform(X_test)



from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
#regressor = LinearRegression() 
#print dataset.isnull().any()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#print('Coefficients: \n', regressor.coef_)
# The mean squared error
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))