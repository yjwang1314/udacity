# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 10:05:56 2018

@author: Jun Wang
"""

import xml.etree.cElementTree as ET
# import pprint
import re
import codecs
import json
from audit import update_name


lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]
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
            "Dong": "East",
            "Xi": "West",
            "Nan": "South",
            "Bei": "North",
            "N": "North",
            "S": "South", 
            "E": "East",
            "W": "West"
            }

def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way" :
        # YOUR CODE HERE
        created = {}
        node_refs = []
        
        attrib = element.attrib
        for field in CREATED:
            created[field] = attrib[field]
        node['created'] = created
        if 'lat' in attrib.keys():
            node['pos'] = [float(attrib['lat']), float(attrib['lon'])]
        node['type'] = element.tag
        if 'visible' in attrib.keys():
            node['visible'] = attrib['visible']
        node['id'] = attrib['id']
        
        if len(element.getchildren()):
            address = {}
            for elem in element.getchildren():
                if elem.tag == 'nd':
                    node_refs.append(elem.attrib['ref'])
                    node['node_refs'] = node_refs
                if elem.tag == 'tag':
                    k = elem.attrib['k']
                    v = elem.attrib['v']
                    if not len(re.findall(problemchars, k)):
                        addr_k = re.findall(re.compile('^addr\:(\D+)'), k)
                        if len(addr_k):
                            addr_k = addr_k[0].split(':')
                            if len(addr_k) == 1:
                                better_v = update_name(v, mapping)
                                address[addr_k[0]] = better_v
                                node['address'] = address
                        elif k.find(':') == -1 and k == 'name':
                            node[k] = update_name(v, mapping)
                        elif k.find(':') == -1:
                            node[k] = v
                        else:
                            pass
        return node
    else:
        return None

def process_map(file_in, pretty = False):
    # You do not need to change this file
    file_out = "{0}.json".format(file_in)
    data = []
    counter = 0 #added counter to show status when creating json file
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            counter += 1
            print counter
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def test():
    data = process_map('map.osm', True)
    #pprint.pprint(data)

if __name__ == "__main__":
    test()