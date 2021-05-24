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

#download data at https://www.kaggle.com/blastchar/telco-customer-churn

def bank_deep_learning_model():
    
    bank_data= pd.read_csv()
    print(bank_data)

    bank_data=bank_data.drop('CustomerId', axis='columns')
    bank_data=bank_data.drop('Surname', axis='columns')

    bank_data=bank_data.drop('RowNumber', axis='columns')

    bank_data

    bank_data['Gender'].replace({'Female':1, 'Male':0}, inplace=True)

    bank_data

    bank_data=pd.get_dummies(data=bank_data, columns=['Geography'])
    bank_data

    col_to_scale=['CreditScore','Age','Tenure','EstimatedSalary']

    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()

    bank_data[col_to_scale]=scaler.fit_transform(bank_data[col_to_scale])

    bank_data

    X=bank_data.drop('Exited', axis='columns')
    y=bank_data['Exited']

    import tensorflow as tf
    from tensorflow import keras

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

    model=keras.Sequential([
        keras.layers.Dense(100, input_shape=(X.shape[1],), activation='relu'),
        keras.layers.Dense(500, activation='sigmoid'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(X_train,y_train, epochs=100)

    model.evaluate(X_test,y_test)

    yp=model.predict(X_test)

    y_pred=[]

    for i in yp:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    y_pred[:10]  

    y_test[:10]

    print(classification_report(y_test,y_pred))

    return 

bank_deep_learning_model()
