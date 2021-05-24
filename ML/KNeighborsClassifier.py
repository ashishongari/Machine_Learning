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
# import talib
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
import glob

# nifty = get_history(symbol='NIFTY 50', start=date(2010,1,1), end=date.today(), index=True)


path =r''
filenames = glob.glob(path + "/*.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))

nifty = pd.concat(dfs, ignore_index=True)

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

X=nifty[['Open', 'High', 'Low', 'Close', 'close_open', 'high_low']].values
y=nifty['Result'].values

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)

print(nifty)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

Ks = 50
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

plt.figure(figsize=(10,6))
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 



