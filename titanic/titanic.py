#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:34:30 2019

@author: zoeyyang
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import csv
import tensorflow as tf
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier


def loadtrain():
    data=pd.read_csv("./train.csv")
    y=data["Survived"]
    return data,y  #(819,12)
def loadtest():
    data=pd.read_csv("./test.csv")
    return data

def exploration(data):
    #data["Sex"]
    #pclass=data["Pclass"]
    no_survived=data["Survived"].sum()
    print(no_survived)
    
    survived_df = data.groupby("Sex").sum().Survived #sum up column value 
    print(survived_df )
    gender_df = data.groupby("Sex").count() #count column no.
    print(survived_df/no_survived )
    
    pclass_df=data.groupby("Pclass").count()
    print(pclass_df.PassengerId)
    survived2_df =data.groupby("Pclass").sum().Survived
    print(survived2_df/no_survived)
    
def onehotencoding(data):
    encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")
    labelencoder =  LabelEncoder()
    
    arr_pclass=np.mat(data["Pclass"], dtype=int).T
    pclass_encode=encoder.fit_transform(arr_pclass)
    for i in range(np.shape(pclass_encode)[1]):
        data["penc_"+str(i+1)]=pclass_encode[:,i]#1,2,3=pclass 1,2,3     
    
    data["Embarked"].replace({'S':1, 'C':2, 'Q':3, np.nan:4}, inplace=True)
    
    arr_embarked=np.mat(data["Embarked"]).T
    embarked_encode=encoder.fit_transform(arr_embarked)
    for i in range(np.shape(embarked_encode)[1]):
        data["emkenc_"+str(i+1)]=embarked_encode[:,i]#enbard
    del(data["Embarked"])
    if("emkenc_4" in data.columns):
        del(data["emkenc_4"]) #get rid of the nan then it would just 000
        
    data["Sex"].replace({'female':1, 'male':2}, inplace=True) 
    arr_sex=np.mat(data["Sex"]).T
    sex_encode=encoder.fit_transform(arr_sex)
    for i in range(np.shape(sex_encode)[1]):
        data["sexenc_"+str(i+1)]=sex_encode[:,i]#enbarkmd
    del(data["Sex"])
    
    
    arr_title=np.mat(data["Title"]).T
    
    arr_title=labelencoder.fit_transform(arr_title)
    arr_title = np.mat(arr_title, dtype=int).T
    title_encode=encoder.fit_transform(arr_title)
    for i in range(np.shape(title_encode)[1]):
        data["titleenc_"+str(i+1)]=title_encode[:,i]
    del(data["Title"])

#for featurenomalization, should use the mean in train to mean norma, if you want to write your own norm
#function
def featurenormalization(data):
    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(data)
    
    return imp
    

def getfeature_cat(data):
    data["Ageclass"]=data.Age*data.Pclass
    data["Ageparch"]=data.Age*data.Parch
    data["Age"].replace({np.nan: np.mean(data["Age"])}, inplace=True)
    data["Fare"].replace({np.nan: np.mean(data["Fare"])}, inplace=True)
    X = data[['Age','Ageclass','Ageparch','Fare','SibSp','Parch','penc_1', 'penc_2', 'penc_3', 'emkenc_1', 'emkenc_2', 'emkenc_3','sexenc_1','sexenc_2','titleenc_1','titleenc_2','titleenc_3','titleenc_4']]
    return X

      
def predict(classifier, imp):
    testdata=loadtest()
    gettitle(testdata)
    onehotencoding(testdata) 
    X_test = getfeature_cat(testdata)
    X_test = imp.transform(X_test)
    result = classifier.predict(X_test)
    
    test_df=pd.DataFrame(columns=["PassengerId","Survived"])
    test_df["PassengerId"]=testdata["PassengerId"]
    test_df["Survived"]=result
    test_df.to_csv("submission.csv", index=False)   
    
    
def gettitle(data):
    title={'Mrs': 'Mrs',
           'Mr': 'Mr',
           'Master': 'Mr',
           'Miss':'Miss', 
           'Major':'Mr',
           'Rev':'Mr',
           'Dr':'Dr',
           'Ms':'Miss',
           'Mlle':'Miss',
           'Col':'Mr',
           'Capt':'Mr', 
           'Mme':'Mrs', 
           'Countess':'Mrs',
           'Don':'Mr',
           'Dona':'Mrs',
           'Jonkheer':'Mr',
           'Lady':'Miss',
           'Sir':'Mr'}
    
    ##!!! need to learn how the list comprehension works
    
    titles = [title[el].lower() for person in data.Name for substr in person.replace(".","").lower().split(" ") for el in title.keys() if el.lower() == substr]
    
    data["Title"] = titles
    

def main():
    train_data,y_train=loadtrain()
    gettitle(train_data)
    
    onehotencoding(train_data) 
#    featurenormalization(data)
    X_train = getfeature_cat(train_data)
    
    imp = featurenormalization(X_train)
    X_train = imp.transform(X_train)
    #   classifier=LogisticRegression(C=0.1)
    classifier=RandomForestClassifier(n_estimators=1000)
    classifier.fit(X_train,y_train)
    
    predict(classifier, imp)
    #    
    #    testdata=loadtest()
    #    onehotencoding(testdata) 
    #    x_test=getfeature(testdata)
    #    predictions=classifier.predict(x_test)
    #    trainper=y_train.sum()/np.shape(y_train)[0]
    #    per=predictions.sum()/np.shape(predictions)[0]
    #classifier = LinearRegression()
    cv_results = cross_validate(classifier,X_train,y_train, cv=LeaveOneOut(),return_train_score=True, n_jobs=-1)
    
    print(cv_results['test_score'].mean(), cv_results['train_score'].mean())
    #    print(per,"train:",trainper)
    #    print(data.describe())
main()      