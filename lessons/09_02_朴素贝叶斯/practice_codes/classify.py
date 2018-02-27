# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:48:46 2018

@author: Jun Wang
"""

def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)


    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    
    accuracy = clf.score(features_test, labels_test)
    # accuracy = sum(pred == labels_test) / float(len(pred)) # 计算准确性的另一种方法
    return accuracy