# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:39:42 2018

@author: Jun Wang
"""

import xml.etree.cElementTree as ET
import pprint

def count_tags(filename):
    tags_list = []
    for event, elem in ET.iterparse(filename):
        tags_list.append(elem.tag)
    tags = list(set(tags_list))
    tags = dict(map(lambda x: [x, tags_list.count(x)], tags))
    return tags

def test():

    tags = count_tags('map.osm')
    pprint.pprint(tags)    

if __name__ == "__main__":
    test()