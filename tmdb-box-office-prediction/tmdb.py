#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:00:37 2019

@author: zoeyyang
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import csv
import json
import tensorflow as tf
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn import linear_model
from sklearn.preprocessing import Imputer  #handler missing data
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

def loadtrain():
    data=pd.read_csv("./train.csv")
    y=data["revenue"]
    del(data["revenue"])
    return data,y
def loadtest():
    data=pd.read_csv("./test.csv")
    return data
    
def exploration(data):
    #data["Sex"]
    #pclass=data["Pclass"]
    
    language_df = data.groupby("original_language").count() #sum up column value 
    
    collection_df = data.groupby("belongs_to_collection").count() #count column no.
    
def preprocessjson(jstring):
    jstring = jstring.replace("\'", "\"")
    jstring = jstring.replace(" n\"", "and")
    jstring = jstring.replace("\"s", "s")
    jstring = jstring.replace("None", "\"None\"")
    return jstring


def onehotencoding(data):
    encoder = OneHotEncoder(sparse=False,handle_unknown="ignore")
    labele =  LabelEncoder()
    
    arr_title=labelencoder.fit_transform(arr_title)
    arr_title = np.mat(arr_title, dtype=int).T
    title_encode=encoder.fit_transform(arr_title)
    for i in range(np.shape(title_encode)[1]):
        data["titleenc_"+str(i+1)]=title_encode[:,i]
    del(data["Title"])
    
def uniqueids(col):
    dsuperall = []
    for row in col:
        try:
            dsuperall += extractid(row)
        except:
            continue #deal with nan
    return sorted(set(dsuperall))
    
def extractid(jsondata):
    try:
        dictdata = jsondata[1:-1]
        d2 = dictdata.replace('},','}*')
        d2 =d2.split("*")
        dall=[]
        for d3 in d2:
            d4=json.loads(preprocessjson(d3))
            dall=dall+[d4['id']]###!!!! i need to put bracket on the id
        #print(dall)
        return dall
    except TypeError:  #deal with nan
        #return jsondata
        return []
    
def formmatrix(data,colname,uniqid):
    n=np.shape(uniqid)[0]
    col=data[colname]
    m=np.shape(col)[0]
    matrix=np.zeros((m,n))
    
    for i,row in enumerate(col):
        dictrow=extractid(row)
        for j, el in enumerate(uniqid):
            if el in dictrow:
                matrix[i,j]=1   
    return matrix
def getfeature(data):
    X=data[['popularity','budget','runtime']]
    
    return X

def featurenormalization(data):
    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(data)
    
    return imp        

data,y=loadtrain()
exploration(data)
#data['belongs_to_collection_id'] = data['belongs_to_collection'].apply(extractid)
uniqid=uniqueids(data.genres)

formmatrix(data,"genres",uniqid)
getfeature(data)

def predict(classifier, imp, uniqid):
    testdata=loadtest()   
    matrix=formmatrix(testdata,"genres",uniqid)
    print(np.shape(matrix))
    
    X_test = getfeature(testdata)
    X_test = imp.transform(X_test)
    
    X_test=preprocessing.scale(X_test)
    
    X_test=np.concatenate((X_test, matrix), axis=1)
    result = classifier.predict(X_test)
    
    test_df=pd.DataFrame(columns=["id","revenue"])
    test_df["id"]=testdata["id"]
    test_df["revenue"] = np.maximum(result, 0)
    test_df.to_csv("submission.csv", index=False)   
    
def main():
    data,y_train=loadtrain()
    #data['belongs_to_collection_id'] = data['belongs_to_collection'].apply(extractid)
    uniqid=uniqueids(data.genres)    
    matrix=formmatrix(data,"genres",uniqid)
    
    print(matrix)
    X_train=getfeature(data)
    imp = featurenormalization(X_train)
    X_train = imp.transform(X_train)
    X_train=preprocessing.scale(X_train)
    
    X_train=np.concatenate((X_train, matrix), axis=1)
    
    classifier=linear_model.Ridge()
    classifier.fit(X_train,y_train)
    predict(classifier,imp, uniqid)   
    
    #    predict(classifier, imp)
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
    
    import matplotlib.pyplot as plt
    plt.scatter(range(len(y_train)),classifier.predict(X_train))
    plt.scatter(range(len(y_train)), y_train)
    plt.show()
    
main()  

