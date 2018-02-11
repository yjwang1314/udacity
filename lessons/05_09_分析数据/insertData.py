# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:30:17 2018

@author: Jun Wang
"""

import csv
from pymongo import MongoClient


filename = 'cities_1.csv'
data = {}
data_list = []
n = 0

with open(filename,'rb') as f:
    reader = csv.DictReader(f)
    for i in range(3):
        reader.next()
    for row in reader:
        
        isPartOf = row['isPartOf_label']
        if '{' in isPartOf:
            isPartOf = isPartOf[1:-1].split('|')
            
        name = row['name']
        if '{' in name:
            name = name[1:-1].split('|')[0]
        
        data['_id'] = n
        data['name'] = row['name']
        data['country'] = row['country'].split('/')[-1]
        data['isPartOf'] = isPartOf
        data['population'] = row['populationTotal']
        # db.uda.insert(data)
        n += 1
        data_list.append(data)
        
        
client = MongoClient('45.62.113.172',27017)
db = client.db_news
db.authenticate()
        
db.uda.insert_many(data_list)