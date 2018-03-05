#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

nb_poi = sum(map(lambda x: x['poi'], enron_data.values())) # 嫌疑人数量
enron_data['PRENTICE JAMES']['total_stock_value']
enron_data['COLWELL WESLEY']['from_this_person_to_poi']
enron_data['SKILLING JEFFREY K']['exercised_stock_options']

enron_data['SKILLING JEFFREY K']['total_payments'] # CEO
enron_data['LAY KENNETH L']['total_payments'] # Founder
enron_data['FASTOW ANDREW S']['total_payments'] # CFO

nb_sal = sum(map(lambda x: x['salary'] != 'NaN', enron_data.values())) # salary数量          
nb_email = sum(map(lambda x: x['email_address'] != 'NaN', enron_data.values())) # email数量         

nb_pay = sum(map(lambda x: x['total_payments'] == 'NaN', enron_data.values()))
float(nb_pay) / len(enron_data)

nb_pay = sum(map(lambda x: x['total_payments'] == 'NaN' and x['poi'], enron_data.values()))

nb_stk = sum(map(lambda x: x['total_stock_value'] == 'NaN' and x['poi'], enron_data.values()))
