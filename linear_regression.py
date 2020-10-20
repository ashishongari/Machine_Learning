import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas_datareader as pdr
import datetime 
import yfinance as yf
from nsepy import get_history
from datetime import date
import datetime as dt
from nsetools import Nse
import talib
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import scipy.optimize as opt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score


#Data Source
tickers_list = ['^NSEBANK','^NSEI']
                               
#DATA STORAGE
data= pd.DataFrame(columns=tickers_list)

#Feth the data
for ticker in tickers_list:
    data[ticker]=pdr.get_data_yahoo(ticker, start=datetime.datetime(2015, 1, 1), end=datetime.datetime(2020,5,29))['Adj Close']

data=data.dropna()
print(data)


#LINEAR REGRESSION MODEL

plt.figure(figsize=(18,10))
plt.style.use('seaborn-darkgrid')
plt.scatter(data['^NSEBANK'],data['^NSEI'], color="black")
plt.title("Scatter Plot of Variables")

x=data['^NSEBANK'].values.reshape(-1,1)
y=data['^NSEI'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()  
regressor.fit(x_train, y_train)

print("Intercept of model is :",str(regressor.intercept_))
print("Slope of Model is :" ,str(regressor.coef_))

y_pred = regressor.predict(x_test)

predata = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
predata['New Data']=predata['Predicted']+228.96


plt.style.use('seaborn-darkgrid')
plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red')
plt.title("Prediction")
plt.show()

print('R Squared Value is', str(r2_score(y_test,y_pred)*100))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

predata['Error Percent']=(predata['New Data']/predata['Actual']-1)*100
predata['Error Percent'].describe()
