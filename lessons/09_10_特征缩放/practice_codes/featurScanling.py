# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 11:41:21 2018

@author: Jun Wang
"""

""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):
    
    try: 
        arr = map(lambda x: (float(x) - min(arr)) / (max(arr) - min(arr)), arr)
        return arr
    except:
        print 'divide-by-zero error'
    
    

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
# data = [10, 10, 10]
print featureScaling(data)

# sklearn way
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = map(float , data)
rescaled_data = scaler.fit_transform(data)
print rescaled_data
