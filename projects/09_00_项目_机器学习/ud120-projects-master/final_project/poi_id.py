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

### Task 3: Create new feature(s)

## 3.1 kbest score features
from feature_selecting import Select_K_Best
k_best_features = Select_K_Best(data_dict, features_list, 18)

## 3.2 create new feature and check
from feature_creation import CreatePoiEmailRatio
CreatePoiEmailRatio(data_dict, features_list)
k_best_features = Select_K_Best(data_dict, features_list, 5)
k_best_features_list = map(lambda x: x[0], k_best_features)
k_best_features_list.insert(0, 'poi')
features_list = k_best_features_list

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### try different classifiers, parameters, to find out the best classifer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, \
                            precision_score, recall_score, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import nearest_centroid
from sklearn.svm import SVC

## split data to train and test
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

## DT
clf = DecisionTreeClassifier()
param_grid = {'min_samples_split': [2, 5, 10, 30]}
clf = GridSearchCV(clf, param_grid)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
best_param = clf.best_params_ # best parameter

result = {}
for v in param_grid['min_samples_split']:
    clf = DecisionTreeClassifier(min_samples_split=v)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = clf.score(features_test, labels_test)
    precision = precision_score(labels_test, pred)
    recall = recall_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)
    result[v] = {'accuracy': accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f1': f1}
    
tuples = zip(param_grid['min_samples_split'], map(lambda x: x['f1'], result.values()))
best_result = sorted(tuples, key=lambda x: x[1], reverse=True)


print classification_report(labels_test, pred)
print confusion_matrix(labels_test, pred)


## PCA + DT
'''
estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
pipe = Pipeline(estimators)
parameters = {'reduce_dim__n_components': [None, 2, 3, 4, 5], 
              'clf__min_samples_split': [3, 4, 5, 6, 7, 8]}
clf = GridSearchCV(pipe, parameters, scoring='f1')
'''

'''
clf = GaussianNB()
clf.fit(features_train, labels_train)
clf.score(features_test, labels_test)
pred = clf.predict(features_test)
'''


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()


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
https://github.com/nehal96/Machine-Learning-Enron-Fraud
https://github.com/yijigao/Enron_project
'''