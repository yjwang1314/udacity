# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 15:54:35 2018

@author: Jun Wang
"""

import codecs
import csv
import json
import pprint
import re

DATAFILE = 'arachnid.csv'
FIELDS ={'rdf-schema#label': 'label',
         'URI': 'uri',
         'rdf-schema#comment': 'description',
         'synonym': 'synonym',
         'name': 'name',
         'family_label': 'family',
         'class_label': 'class',
         'phylum_label': 'phylum',
         'order_label': 'order',
         'kingdom_label': 'kingdom',
         'genus_label': 'genus'}


def process_file(filename, fields):

    process_fields = fields.keys()
    data = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for i in range(3):
            l = reader.next()

        for line in reader:
            # YOUR CODE HERE
            # print line
            # pass
            data_dict = {}
            classification = {'kingdom':'',\
                              'family':'',\
                              'order':'',\
                              'phylum':'',\
                              'genus':'',\
                              'class':''}
            # data_dict['classification'] = classification
            
            for field in process_fields:
                
                if field == 'rdf-schema#label':
                    # 处理label中的小括号
                    label = line[field]
                    find_bracket = lambda x: re.findall(re.compile('\(\w+\)+'), x)
                    label_bracket = find_bracket(label)
                    if len(label_bracket):
                        for bracket in label_bracket:
                            label = label.replace(bracket,'').strip()
                    if label == 'NULL':
                        label = None
                    data_dict[fields[field]] = label
                             
                elif field == 'name':
                    # 如果“name”为“NULL”，或包含非字母数字字符，将其设为和“label”相同的值。
                    name = line[field]
                    if name == 'NULL' or len(re.findall(re.compile('\W'), name)):
                        name = label
                    data_dict[fields[field]] = name
                             
                elif field == 'synonym':
                    # 如果“synonym”中存在值，应将其转换为数组（列表）
                    synonym = line[field]
                    if synonym == 'NULL': 
                        synonym = None
                    else:
                        synonym = parse_array(synonym)
                    data_dict[fields[field]] = synonym
                             
                else:
                    field_value = line[field]
                    # 如果字段的值为“NULL”，将其转换为“None”
                    if field_value == 'NULL':
                        field_value = None
                    # 删掉所有字段前后的空格（如果有的话）
                    elif type(field_value) == str:
                        field_value = field_value.strip()
                    
                    if fields[field] in classification.keys():
                        classification.update({fields[field] : field_value})
                        data_dict['classification'] = classification
                    else:
                        data_dict[fields[field]] = field_value
                                        
            data.append(data_dict)
                    
            
    return data


def parse_array(v):
    if (v[0] == "{") and (v[-1] == "}"):
        v = v.lstrip("{")
        v = v.rstrip("}")
        v_array = v.split("|")
        v_array = [i.strip() for i in v_array]
        return v_array
    return [v]


def test():
    data = process_file(DATAFILE, FIELDS)
    print "Your first entry:"
    pprint.pprint(data[0])
    first_entry = {
        "synonym": None, 
        "name": "Argiope", 
        "classification": {
            "kingdom": "Animal", 
            "family": "Orb-weaver spider", 
            "order": "Spider", 
            "phylum": "Arthropod", 
            "genus": None, 
            "class": "Arachnid"
        }, 
        "uri": "http://dbpedia.org/resource/Argiope_(spider)", 
        "label": "Argiope", 
        "description": "The genus Argiope includes rather large and spectacular spiders that often have a strikingly coloured abdomen. These spiders are distributed throughout the world. Most countries in tropical or temperate climates host one or more species that are similar in appearance. The etymology of the name is from a Greek name meaning silver-faced."
    }

    assert len(data) == 76
    assert data[0] == first_entry
    assert data[17]["name"] == "Ogdenia"
    assert data[48]["label"] == "Hydrachnidiae"
    assert data[14]["synonym"] == ["Cyrene Peckham & Peckham"]

if __name__ == "__main__":
    test()