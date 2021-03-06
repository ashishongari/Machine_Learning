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
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

nifty = get_history(symbol='NIFTY 50', start=date(2010,1,1), end=date.today(), index=True)

niftyret=nifty['Close'].pct_change()
nifty['Return']=niftyret
nifty=nifty[['Open','High','Low','Close','Return']]
nifty=nifty.dropna()
nifty['close_open']=nifty['Open'].sub(nifty['Close'])
nifty['high_low']=nifty['High'].sub(nifty['Low'])

nifty.loc[nifty['Return'] >0,'Result']=1
nifty.loc[nifty['Return'] <0,'Result']=0
nifty['Result']=nifty['Result'].shift(-1)

nifty=nifty[['Open', 'High', 'Low', 'Close', 'close_open', 'high_low','Result']]
nifty=nifty.dropna()
# print(nifty.head())

X=nifty[['Open', 'High', 'Low', 'Close', 'close_open', 'high_low']].values
y=nifty['Result'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 

yhat = clf.predict(X_test)
print(yhat[0:5])

from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test,yhat))