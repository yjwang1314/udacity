#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(kernel='linear')

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# training time: 226.35 s

t0 = time()                             
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
# predict time: 23.974 s

accuracy = clf.score(features_test, labels_test)
print "predict accuracy:", round(accuracy, 4)
# predict accuracy: 0.9841
#########################################################

# 加快算法速度的一种方式是在一个较小的训练数据集上训练它。这样做换来的是准确率几乎肯定会下降。
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
# training time: 0.124 s

t0 = time()                             
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"
# predict time: 1.295 s

accuracy = clf.score(features_test, labels_test)
print "predict accuracy:", round(accuracy, 4)
# predict accuracy: 0.8845

# 这两行有效地将训练数据集切割至原始大小的 1%，丢弃掉 99% 的训练数据。但准确率也不低。
# 如果速度是一个主要考虑因素（对于许多实时机器学习应用而言确实如此），并且如果牺牲一些准确率可加快你的训练/预测速度，则你可能会想这样做。

#########################################################
# 更复杂的内核能提高准确率？ rbf
def test_rbf(kernal, c):
    clf = SVC(kernel='rbf', C=c)
    
    t0 = time()
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    # training time: 0.156 s
    
    t0 = time()                             
    pred = clf.predict(features_test)
    print "predict time:", round(time()-t0, 3), "s"
    # predict time: 1.373 s
    
    accuracy = clf.score(features_test, labels_test)
    print "predict accuracy:", round(accuracy, 4)
    # predict accuracy: 0.616
    
    return pred, accuracy

test_rbf('rbf', 1.0)

# 尝试不同C值
c_list = [10.0, 100., 1000., 10000.]
map(lambda x: test_rbf('rbf', x), c_list)
'''
[0.61604095563139927,
 0.61604095563139927,
 0.82138794084186573,
 0.89249146757679176]
'''

#########################################################
# 优化后的 RBF 与线性 SVM：准确率
features_train, features_test, labels_train, labels_test = preprocess()
pred, acry = test_rbf('rbf', 10000)
# training time: 136.42 s
# predict time: 13.201 s
# predict accuracy: 0.9909

map(lambda x: pred[x], [10,26,50])
print 'number of email written by Chris:', sum(pred == 1)
