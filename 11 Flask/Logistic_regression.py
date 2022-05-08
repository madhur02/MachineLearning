# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:35:03 2020

@author: MaJain
"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

train = pd.read_csv(r'C:\Users\majain\Desktop\DataScience\DataScience by Madhur\04 Machine Learning\Logistic Regression\titanic_train.csv')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1 , inplace = True)
train['Embarked'].fillna('S', inplace = True)

sex = pd.get_dummies(train['Sex'],drop_first=True)
embarked = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,sex ,embarked], axis = 1)


train.drop(['Sex','Embarked' ],axis = 1 , inplace = True)
x = train.drop('Survived', axis = 1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
lr = LogisticRegression()
lr.fit(X_train , y_train)
predict = lr.predict(X_test)

print(classification_report(y_test , predict))
print(confusion_matrix(y_test , predict))


## pickling the model
my_file = open('logistic.pickle', 'wb')
pickle.dump(lr,my_file)
my_file.close()