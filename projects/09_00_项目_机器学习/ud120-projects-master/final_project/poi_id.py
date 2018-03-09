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

## 3.1 select kbest features
from feature_selecting import Select_K_Best
k_best_features = Select_K_Best(data_dict, features_list, 15)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


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
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

'''
https://github.com/nehal96/Machine-Learning-Enron-Fraud
https://github.com/yijigao/Enron_project
'''