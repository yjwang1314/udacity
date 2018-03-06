#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

# show outliers
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# show outliers keys
outlier_bonus = data[:,1].max()

for v in data_dict.items():
    if v[1]['bonus'] == outlier_bonus:
        break
key = v[0]

# move outliers
data_dict.pop(key, 0)
data = featureFormat(data_dict, features)

# show outliers
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

# get outliers keys
outlier_data = data[data[:,1] > 5000000]
outlier_data = outlier_data[outlier_data[:,0] > 1000000]

keys = []
for v in data_dict.items():
    for out_sal, out_bon in outlier_data:
        if v[1]['bonus'] == out_bon and v[1]['salary'] == out_sal:
            keys.append(v[0])
print keys # two poi, don't move them
