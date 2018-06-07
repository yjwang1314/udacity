# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 16:36:08 2018

@author: Jun Wang
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat
import matplotlib.pyplot as plt


def findOutlier_NaN(data_dict, max_perct=0.9):
    
    nb_nan_dict = {}
    keys = data_dict.keys()
    features = data_dict[keys[0]].keys()
    for key, value in data_dict.iteritems():
        nb_nan = sum(map(lambda x: value[x] == 'NaN', features)) / float(len(features))
        if nb_nan > max_perct:
            nb_nan_dict[key] = nb_nan
    
    return nb_nan_dict    

def findOutlier_visual(data_dict):
    data = featureFormat(data_dict, ['salary', 'bonus'])
    # show outliers
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter( salary, bonus )
    
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()
    
    # show outliers keys
    outlier = data[:,1].max()
    
    for v in data_dict.items():
        if v[1]['bonus'] == outlier:
            break
    key = v[0]
    
    return key

if __name__ == '__main__':
    nb_nan_dict = findOutlier_NaN(data_dict)
    print nb_nan_dict