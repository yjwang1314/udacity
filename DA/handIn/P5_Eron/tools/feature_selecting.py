# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 17:38:09 2018

@author: Jun Wang
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest

def Select_K_Best(data_dict, features_list, k):
    """
    Runs scikit-learn's SelectKBest feature selection algorithm, returns an
    array of tuples with the feature and its score.
    """

    data_array = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data_array)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    tuples = zip(features_list[1:], scores)
    k_best_features = sorted(tuples, key=lambda x: x[1], reverse=True)

    return k_best_features[:k]