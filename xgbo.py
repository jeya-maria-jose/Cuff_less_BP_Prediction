import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression  
import numpy as np  
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('final_clean.csv',names = ['alpha','PIR', 'ptt', 'bpmax' ,'bpmin', 'hrfinal', 'ih', 'il', 'meu', 'j', 'k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac'])

X = dataset[['alpha','PIR', 'ptt','hrfinal', 'ih', 'il', 'meu','j', 'k','l','m','n','o','p','q','r','s']]

y = dataset[['bpmax']]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
x_train=sc_X.fit_transform(X_train)
x_test=sc_X.transform(X_test)



from xgboost import XGBClassifier
regressor = XGBClassifier()
#regressor = LinearRegression() 
#print dataset.isnull().any()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)


#print('Coefficients: \n', regressor.coef_)
# The mean squared error
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



# # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))