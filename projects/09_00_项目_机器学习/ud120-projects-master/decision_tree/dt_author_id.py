#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# training time:  98.976 s

t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
# predict time: 0.075 s

acc = clf.score(features_test, labels_test)
print 'accuracy:', round(acc, 4)
# accuracy: 0.9767

#########################################################

# 通过特征选择加速
len(features_train[0]) # 特征数 3785
# 1% 特征下的准确率为0.9664，但时间减少了很多
