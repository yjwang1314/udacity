#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
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

# from findOutliers import findOutlier_NaN, findOutlier_visual
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
from feature_selecting import Select_K_Best
k_best_features = Select_K_Best(data_dict, features_list, len(features_list)-1)

## 3.2 create new feature and check
from feature_creation import CreatePoiEmailRatio
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
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)

### try different classifiers, parameters, to find out the best classifer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, \
                            precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import nearest_centroid, KNeighborsClassifier

from tester_2 import test_classifier, test_clf_split

## split data to train and test
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## PCA + DT
estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
pipe = Pipeline(estimators)
parameters = {'reduce_dim__n_components': [None, 2, 3, 4, 5], 
              'clf__min_samples_split': [2, 5, 8, 10, 30]}
clf = GridSearchCV(pipe, parameters, scoring='f1')
result_1 = test_clf_split(clf, my_dataset, features_list)
result_2 = test_classifier(clf, my_dataset, features_list, folds = 1000)

## 默认参数
names = ["Naive Bayes", "Decision Tree", "Nearest Centroid", 
         "Random Forest", "AdaBoost"]
classifiers = [GaussianNB(),
               DecisionTreeClassifier(),
               nearest_centroid.NearestCentroid(),
               RandomForestClassifier(),
               AdaBoostClassifier(base_estimator=DecisionTreeClassifier())]

result_1 = {}	
for name, clf in zip(names, classifiers):
    print("feature:", features_list)
    result_1[name] = test_classifier(clf, my_dataset, features_list, folds=1000)

'''    
{'Naive Bayes': {'Accuracy': 0.84300       
                'Precision': 0.48581      
                'Recall': 0.35100 
                'F1': 0.40755     
                'F2': 0.37163},
'Decision Tree': {'Accuracy': 0.80154       
                'Precision': 0.36449      
                'Recall': 0.39000 
                'F1': 0.37681     
                'F2': 0.38462},
'Nearest Centroid': {'Accuracy': 0.83338       
                'Precision': 0.45765      
                'Recall': 0.44850 
                'F1': 0.45303     
                'F2': 0.45030},
'Random Forest': {'Accuracy': 0.86392       
                'Precision': 0.60990      
                'Recall': 0.32050 
                'F1': 0.42019     
                'F2': 0.35410},
'AdaBoost': {'Accuracy': 0.80100       
                'Precision': 0.36279      
                'Recall': 0.38800 
                'F1': 0.37497     
                'F2': 0.38268}}   
'''             
    
## 自动调节参数
parameters = {"Naive Bayes":{},
				  "Decision Tree":{"max_depth": range(5,15),
								   "min_samples_leaf": range(1,5),
                           "min_samples_split": [2, 5, 8, 10, 30]},
				  "Nearest Centroid":{"shrink_threshold": [None, 0.2, 0.6, 0.8, 1]},
				  "Random Forest":{"n_estimators": range(2, 5),
								   "min_samples_split": [2, 5, 8, 10, 30],
								   "max_depth": range(2, 15),
								   "min_samples_leaf": range(1, 5),
								   "random_state": [0, 10, 23, 36, 42],
								   "criterion": ["entropy", "gini"]},
				  "AdaBoost":{"n_estimators": range(2, 5),
							  "algorithm":("SAMME", "SAMME.R"),
							  "random_state":[0, 10, 23, 36, 42]}}
print("feature:", features_list)
for name, clf in zip(names, classifiers):
    grid_clf = GridSearchCV(clf, parameters[name])
    grid_clf.fit(features_train, labels_train)
    print name
    print (grid_clf.best_params_)
    
## 再一次检验模型效果
classifiers = [GaussianNB(),
               DecisionTreeClassifier(max_depth=6, min_samples_leaf=1,
                                      min_samples_split=5),
               nearest_centroid.NearestCentroid(shrink_threshold=None),
               RandomForestClassifier(min_samples_leaf=1, n_estimators=2, 
                                      random_state=23, criterion='entropy', 
                                      min_samples_split=2, max_depth=6),
               AdaBoostClassifier(n_estimators=2, random_state=0, 
                                  algorithm='SAMME', 
                                  base_estimator=DecisionTreeClassifier())]

result_2 = {}		
for name, clf in zip(names, classifiers):
    print("feature:", features_list)
    result_2[name] = test_classifier(clf, my_dataset, features_list, folds=1000)
    
'''    
{'Naive Bayes': {'Accuracy': 0.84300       
                'Precision': 0.48581      
                'Recall': 0.35100 
                'F1': 0.40755     
                'F2': 0.37163},
'Decision Tree': {'Accuracy': 0.82246       
                'Precision': 0.38841      
                'Recall': 0.26800 
                'F1': 0.31716     
                'F2': 0.28571},
'Nearest Centroid': {'Accuracy': 0.83338       
                'Precision': 0.45765      
                'Recall': 0.44850 
                'F1': 0.45303     
                'F2': 0.45030},
'Random Forest': {'Accuracy': 0.85123       
                'Precision': 0.53466      
                'Recall': 0.25450 
                'F1': 0.34485     
                'F2': 0.28429},
'AdaBoost': {'Accuracy': 0.80146       
                'Precision': 0.36619      
                'Recall': 0.39750 
                'F1': 0.38120     
                'F2': 0.39082}}   
''' 

features_list = k_best_features_list[:6]
clf = nearest_centroid.NearestCentroid(shrink_threshold = None)
result_3 = test_classifier(clf, my_dataset, features_list, folds=1000)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()


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

dump_classifier_and_data(clf, my_dataset, features_list)

'''
https://github.com/supernova16/DAND-P5-Machine-Learning
https://github.com/nehal96/Machine-Learning-Enron-Fraud
https://github.com/yijigao/Enron_project
http://nbviewer.jupyter.org/github/DariaAlekseeva/Enron_Dataset/blob/master/Enron%20POI%20Detector%20Project%20Assignment.ipynb
https://github.com/MarcCollado/enron
'''
