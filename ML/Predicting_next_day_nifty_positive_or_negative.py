import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas_datareader as pdr
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from datetime import date
import tensorflow as tf
from tensorflow import keras
from nsepy import get_history

"""
Predict if next trading day on Nifty will be positive or Neagtibve
"""

class index_data:
    
    def __init__(self, ticker):
        self.ticker=ticker
        
    def historical_data(self):
        return get_history(symbol=self.ticker, start=date(2010,1,1), end=date.today(), index=True)
    
nifty_data=index_data('NIFTY 50')
nifty_data_df=nifty_data.historical_data()
# print(nifty_data_df)


def Deep_Nifty():
    
    nifty_data_df_copy=nifty_data_df
    nifty_data_df_copy=nifty_data_df_copy[['Open','High','Low','Close']]
    nifty_data_df_copy['pct_change']=nifty_data_df_copy['Close'].pct_change()

    nifty_data_df_copy['signal']=np.where( nifty_data_df_copy['pct_change'] >  0 ,1,0)
    nifty_data_df_copy['signal']=nifty_data_df_copy['signal'].shift(-1)

    nifty_data_df_copy=nifty_data_df_copy[['Open','High','Low','Close','signal']]
    nifty_data_df_copy=nifty_data_df_copy.dropna()

    X=nifty_data_df_copy[['Open','High','Low','Close']].values
    y=nifty_data_df_copy[['signal']].values

    iteration=5000
    learning_rate=0.01
    num_of_neurons=20
    lambd=0.2

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    y_train=y_train.T
    y_test=y_test.T

    m_train=x_train.shape[0]
    m_test=x_test.shape[0]
    num_px=x_train.shape[1]

    train_x_flatten=x_train.reshape(x_train.shape[0], -1).T
    test_x_flatten = x_test.reshape(x_test.shape[0], -1).T

    train_x = train_x_flatten
    test_x = test_x_flatten

    m=train_x.shape[0]
    n=train_x.shape[1]

    cost_array=[]

    W=np.zeros((num_of_neurons,m))
    b=np.zeros((num_of_neurons,1))

    for i in range(iteration):

        Z=np.dot(W,train_x) + b

        A= 1/(1+np.exp(-Z))
        dZ=A-y_train

        L2_regularization_cost= np.sum(np.square(W)) + (lambd/(2*n))

        cost=(y_train-A)**2
        cost=cost.sum(axis=0)
        cost=cost.reshape(1,cost.shape[0])
        cost=np.sum(cost, axis=1, keepdims=True)*(1/n)
        cost=int(cost)
        cost_array.append(cost)

        A_hat=A.sum(axis=0)

        dW=np.dot(dZ,train_x.T)*(1/n)  + (lambd/n)*W
        db=np.sum(dZ, axis=1, keepdims=True)*(1/n)

        W= W - (dW*learning_rate)
        b= b - (db*learning_rate)



    plt.plot(cost_array)
    plt.title("Gradient Descent")

    pred_data=[]

    for i in range(len(A_hat)):

        if A_hat[i]>0.5:
            pred_data.append(1)
        else:
            pred_data.append(0)


    pred_data=np.array(pred_data)

    pred_data_df=pd.DataFrame(pred_data)

    y_train_df=pd.DataFrame(y_train.T)

    data_df=pd.concat([pred_data_df,y_train_df], axis=1) 

    data_df.columns=['Prediciton','Real']

    data_df['signal']=np.where(data_df['Prediciton']==data_df['Real'],True,False)

    print(data_df['signal'].value_counts(10)*100)
    
    return

Deep_Nifty()

