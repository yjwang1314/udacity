#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    clean_pec = 0.1
    clean_nb = round(len(predictions) * clean_pec, 0)
    sse = (predictions - net_worths)**2
          
    data = np.concatenate((ages, net_worths, sse), axis = 1)
    clean_pos = data[:,2].argsort()[:-clean_nb] # 异常值的配置
    data = data[clean_pos] # 移除异常值
    cleaned_data = list(data)
    
    return cleaned_data

