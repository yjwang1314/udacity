# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:49:16 2018

@author: udacity
"""

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

if __name__ == '__main__':
    accuracy = submitAccuracy()
    print accuracy