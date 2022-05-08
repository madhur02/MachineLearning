# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:23:03 2020

@author: MaJain
"""
import pandas as pd
import pickle 
import numpy as np
from flask import Flask, jsonify, abort, request

app = Flask(__name__)

'''
Below is the route decorator which is used to bind the URL to the function.
'''


@app.route('/')
def hello_world():
    return "Hello Good Evening"

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
    
load_file = open(r'C:\Users\majain\Desktop\DataScience\DataScience by Madhur\04 Machine Learning\Logistic Regression\log_regression.pickle', 'rb')
model = pickle.load(load_file)
def run_ml(train , mdoel = model ):
    train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
    train.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1 , inplace = True)
    train['Embarked'].fillna('S', inplace = True) 
    train['Sex'] = train['Sex'].replace('female',0)
    train['Sex'] = train['Sex'].replace('male',1)
    
    if train['Embarked'].iloc[0] == 'S':
        embark = pd.DataFrame(np.array([0,1]).reshape(1,-1),columns = ['Q','S'])
    elif train['Embarked'].iloc[0] == 'C':
        embark = pd.DataFrame(np.array([0,0]).reshape(1,-1),columns = ['Q','S'])
    else: 
        embark = pd.DataFrame(np.array([1,0]).reshape(1,-1),columns = ['Q','S'])
      
    train = pd.concat([train,embark], axis = 1)
    train.drop(['Embarked'],axis = 1 , inplace = True)
    print(train.head())
    train.dropna(axis=0 , inplace = True)
    prediction = mdoel.predict(train)
    return prediction
    
    
@app.route('/api/complexity', methods = ['POST'])
def get_complexity():
    #content = request.json
    params = {
    'PassengerId': request.json['PassengerId'],
	'Pclass': request.json['Pclass'],
	'Name': request.json['Name'],
	'Sex': request.json['Sex'],
	'Age': request.json['Age'],
	'SibSp': request.json['SibSp'],
	'Parch': request.json['Parch'],
	'Ticket': request.json['Ticket'],
	'Fare': request.json['Fare'],
	'Cabin': request.json['Cabin'],
    'Embarked': request.json['Embarked'],
    }

    df = pd.DataFrame([params])
    print(df.head())

    prediction = run_ml(df)
    print("This is prediction",prediction)
    if prediction[0] == 0:
        result = "Not Survived"
    else: 
         result = "Survived"

    return jsonify( { "This person was": result } ), 200




if __name__ == '__main__':
    app.run(debug = True)