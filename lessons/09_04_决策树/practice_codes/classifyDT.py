# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 17:32:56 2018

@author: Jun Wang
"""
from sklearn import tree 

def classify(features_train, labels_train):
    
    ### your code goes here--should return a trained decision tree classifer
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    
    return clf