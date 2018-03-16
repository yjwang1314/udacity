#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

# data wrangle
from findOutliers import findOutlier_NaN, findOutlier_visual
from feature_creation import CreatePoiEmailRatio

# cross validation
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

# pre-processing
from sklearn.preprocessing import MinMaxScaler
from feature_selecting import Select_K_Best

# machine learning
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import nearest_centroid
from sklearn.svm import SVC

# evaluation
from tester import dump_classifier_and_data
from tester_2 import test_classifier
from sklearn.metrics import classification_report, confusion_matrix, \
                            precision_score, recall_score, f1_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# delect useless features ['others', 'email_address', ]
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
                 'loan_advances', 'bonus', 'deferred_income', 'restricted_stock_deferred', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
print 'Number of person:', len(data_dict)
print 'Number of features:', len(data_dict.values()[0])
print 'Number of POIs:', sum(map(lambda x: x['poi']==True, data_dict.values()))
print 'Number of non-POIs:', sum(map(lambda x: x['poi']==False, data_dict.values()))

### Task 2: Remove outliers
## 2.1 find outliers by outliers data 'NaN' and remove
# key_nan = findOutlier_NaN(data_dict)

## 2.2 find outliers by visualization and remove
# key = findOutlier_visual(data_dict)

## 2.3 remove
data_dict.pop('LOCKHART EUGENE E') # No data available on this person.
data_dict.pop('THE TRAVEL AGENCY IN THE PARK') # Not a person/employee associated with Enron
data_dict.pop('TOTAL') # Summation of everyone's data

print 'Number of person:', len(data_dict)
print 'Number of features:', len(data_dict.values()[0])
print 'Number of POIs:', sum(map(lambda x: x['poi']==True, data_dict.values()))
print 'Number of non-POIs:', sum(map(lambda x: x['poi']==False, data_dict.values()))

### Task 3: Create new feature(s)
## 3.1 kbest score features
k_best_features = Select_K_Best(data_dict, features_list, len(features_list)-1)

## 3.2 create new feature and check
CreatePoiEmailRatio(data_dict, features_list)
k_best_features = Select_K_Best(data_dict, features_list, len(features_list)-1)
k_best_features_list = map(lambda x: x[0], k_best_features)
k_best_features_list.insert(0, 'poi')
features_list = k_best_features_list[:11]

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

## split data to train and test
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

## 默认参数
result_1 = {}	
result_2 = {}	
best_score = 0	
clf_list = []
names = ["Naive Bayes", "Decision Tree", "Nearest Centroid", "SVC", 
         "Random Forest", "AdaBoost"]
classifiers = [GaussianNB(),
               DecisionTreeClassifier(class_weight='balanced',random_state=42),
               make_pipeline(MinMaxScaler(), nearest_centroid.NearestCentroid()),
               make_pipeline(MinMaxScaler(), SVC(class_weight='balanced',random_state=42)),
               RandomForestClassifier(class_weight='balanced',random_state=42),
               AdaBoostClassifier(base_estimator=DecisionTreeClassifier(class_weight='balanced',random_state=42))]

print "feature:", features_list
for name, clf in zip(names, classifiers):
    result_1[name] = test_classifier(clf, my_dataset, features_list, folds=1000)          
    
## 自动调节参数
parameters = {"Naive Bayes":{},
				  "Decision Tree":{"max_depth": range(5,15),
                           "min_samples_leaf": range(1,5),
                           "min_samples_split": [2, 8, 10, 30, 50, 70],
                           'criterion' : ['gini', 'entropy'],
                           'splitter': ['best', 'random']},
				  "Nearest Centroid":{ # Nearest Centroid pipeline
                          "nearestcentroid__shrink_threshold": [None, 0.2, 0.6, 0.8, 1]},
              "SVC":{  # SVC pipeline
                    'svc__kernel' : ['linear', 'rbf', 'poly', 'sigmoid'],
                    'svc__C' : [0.1, 1.0, 10, 100, 1000, 10000],
                    'svc__gamma' : [0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]},
				  "Random Forest":{"n_estimators": range(2, 5),
								   "min_samples_split": [2, 8, 10, 30, 50, 70],
								   "max_depth": range(2, 15),
								   "min_samples_leaf": range(1, 5),
								   "random_state": [0, 10, 23, 36, 42],
								   "criterion": ["entropy", "gini"]},
				  "AdaBoost":{"n_estimators": range(2, 5),
							  "algorithm":("SAMME", "SAMME.R"),
							  "random_state":[0, 10, 23, 36, 42]}}

for name, clf in zip(names, classifiers):
    grid_clf = GridSearchCV(clf, parameters[name], scoring='f1')
    grid_clf.fit(features, labels)
    print '\nname:', name
    print 'best_params:', grid_clf.best_params_
    
    ## 再一次检验模型效果
    # Evaluate every model
    best_estimator_ii = grid_clf.best_estimator_
    best_score_ii = grid_clf.best_score_

    print '------------\nF1 Score:',best_score_ii,'\n'

    result_2[name] = test_classifier(best_estimator_ii, my_dataset, features_list)
    clf_list.append(best_estimator_ii)
    if result_2[name]['f1'] > best_score:
        best_estimator = best_estimator_ii
        best_score = result_2[name]['f1']
#######################################     
features_list = ['poi', 'exercised_stock_options', 'poi_email_ratio']
   
result_3 = {}	
classifiers = clf_list      

for name, clf in zip(names, classifiers):
    result_3[name] = test_classifier(clf, my_dataset, features_list, folds=1000)   
    
'''
## PCA + knn
estimators = [('reduce_dim', PCA()), ('clf', clf)]
pipe = Pipeline(estimators)
param = {'reduce_dim__n_components': [None, 2, 3, 4, 5]}
grid_clf = GridSearchCV(pipe, param, scoring='f1')
result_4 = test_classifier(grid_clf, my_dataset, features_list, folds = 1000)

features_list = ['poi', 'poi_email_ratio']

## PCA + DT
estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
pipe = Pipeline(estimators)
parameters = {'reduce_dim__n_components': [None, 2, 3, 4, 5], 
              'clf__min_samples_split': [2, 5, 8, 10, 30]}
clf = GridSearchCV(pipe, parameters, scoring='f1')
result_4 = test_classifier(clf, my_dataset, features_list, folds = 1000)
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = best_estimator
features_list = ['poi', 'exercised_stock_options', 'poi_email_ratio']
dump_classifier_and_data(clf, my_dataset, features_list)

'''
https://github.com/supernova16/DAND-P5-Machine-Learning
https://github.com/nehal96/Machine-Learning-Enron-Fraud
https://github.com/yijigao/Enron_project
http://nbviewer.jupyter.org/github/DariaAlekseeva/Enron_Dataset/blob/master/Enron%20POI%20Detector%20Project%20Assignment.ipynb
https://github.com/MarcCollado/enron
'''
