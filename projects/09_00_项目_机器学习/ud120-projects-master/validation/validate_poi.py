#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

# split train data and test data, and train, test
features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
accuracy = clf.score(features_test, labels_test)
print 'accuracy:', accuracy # 0.7916666


# search for best parameter
from sklearn.grid_search import GridSearchCV

param_grid = {'min_samples_split': [2, 5, 10, 30],
              'max_depth': [None, 2, 5, 10]}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid)
clf = clf.fit(features_train, labels_train)
accuracy_best = clf.score(features_test, labels_test)
print 'accuracy_best:', accuracy_best # 0.8620689
best_param = clf.best_params_ # best parameter