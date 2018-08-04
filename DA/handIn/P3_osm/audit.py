# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:37:09 2018

@author: Jun Wang
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "map.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE) # 地址最后一个词


expected = ["Street", "Avenue", "Boulevard", "Drive", "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons"]

# UPDATE THIS VARIABLE
mapping = { "St": "Street",
            "St.": "Street",
            "Rd": "Road",
            "Rd.": "Road",
            "road": "Road",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "ave": "Avenue",
            "Sheng": "Province",
            "Shi": "City",
            "Qu": "District",
            "Lu": "Road",
            "Jie": "Street",
            "Xi": "West",
            "N": "North",
            "S": "South", 
            "E": "East",
            "W": "West"
            }


def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types


def update_name(name, mapping):

    # YOUR CODE HERE
    match = re.search(street_type_re,name)
    if match and match.group() in mapping.keys():
        name = re.sub(street_type_re, mapping[match.group()], name)
    return name


def test():
    st_types = audit(OSMFILE)
    # assert len(st_types) == 3
    pprint.pprint(dict(st_types))
    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "=>", better_name

if __name__ == '__main__':
    test()