# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:39:42 2018

@author: Jun Wang
"""

import xml.etree.cElementTree as ET
import pprint
import re

lower = re.compile(r'^([a-z]|_)*$') # 表示仅包含小写字母且有效的标记
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$') # 表示名称中有冒号的其他有效标记
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]') # 表示字符存在问题的标记

def count_tags(filename):
    tags_list = []
    for event, elem in ET.iterparse(filename):
        tags_list.append(elem.tag)
    tags = list(set(tags_list))
    tags = dict(map(lambda x: [x, tags_list.count(x)], tags))
    return tags

def count_tags_test():

    tags = count_tags('map_sample.osm')
    pprint.pprint(tags)    
    

def key_type(element, keys):
    if element.tag == "tag":
        # YOUR CODE HERE
        k = element.attrib['k']
        if len(re.findall(lower, k)):
            keys['lower'] += 1
        elif len(re.findall(lower_colon, k)):
            keys['lower_colon'] += 1
            # print k
        elif len(re.findall(problemchars, k)):
            keys['problemchars'] += 1
        else:
            keys['other'] += 1
            print k
        
    return keys

def process_map(filename):
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys


def test():
    keys = process_map('map_sample.osm')
    pprint.pprint(keys)
    # assert keys == {'lower': 5, 'lower_colon': 0, 'other': 1, 'problemchars': 1}

if __name__ == "__main__":
    test()