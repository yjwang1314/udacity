# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 11:41:53 2018

@author: Jun Wang
"""

def CreatePoiEmailRatio(data_dict, features_list):
    """
    Adds a new feature to the feature list: POI Email Ratio.
    """
    features = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi', 'shared_receipt_with_poi']

    for key in data_dict:
        employee = data_dict[key]
        is_valid = True
        for feature in features:
            if employee[feature] == 'NaN':
                is_valid = False
        if is_valid:
            total_from = employee['from_poi_to_this_person'] + employee['from_messages']
            total_to = employee['from_this_person_to_poi'] + employee['to_messages']
            to_poi_ratio = float(employee['from_this_person_to_poi']) / total_to
            from_poi_ratio = float(employee['from_poi_to_this_person']) / total_from
            receipt_ration = float(employee['shared_receipt_with_poi']) / (total_to + total_from + employee['shared_receipt_with_poi'])
            employee['poi_email_ratio'] = to_poi_ratio + from_poi_ratio + receipt_ration
            employee['from_poi_ratio'] = from_poi_ratio
            employee['to_poi_ratio'] = to_poi_ratio
            employee['shared_with_poi_ratio'] = float(employee['shared_receipt_with_poi']) / (total_from + employee['shared_receipt_with_poi'])
        else:
            employee['poi_email_ratio'] = 'NaN'
            employee['from_poi_ratio'] = 'NaN'
            employee['to_poi_ratio'] = 'NaN'
            employee['shared_with_poi_ratio'] = 'NaN'
    
    map(lambda x: features_list.append(x) ,['poi_email_ratio', 'from_poi_ratio', 'to_poi_ratio', 
                          'shared_with_poi_ratio'])