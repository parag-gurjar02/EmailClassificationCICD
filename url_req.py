# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:57:07 2018

@author: venkatesh.s49
"""
import json
import requests
import pandas as pd

header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

text_data={'Description': 'hi please unlock  my system', 'Subject': 'PO Status', 'Case.Number': 'CASE-14825444', 'is_fyi': 'no', 'Category': 'shipment.status'}

resp = requests.post("http://127.0.0.1:5000/api/model/classify", \
                    data = text_data,\
                    headers= header)
resp.status_code
k=resp.json()

print (k)

# k={'Case.Number': 'CASE-14825444', 'Description': 'hi rfc  my system', 'Category': 'shipment.status', 'is_fyi': 'no', 'Subject': 'PO Status'}
# pd.read_json(k)

# pd.DataFrame.from_dict()
# pd.DataFrame(l, index=[0,1])



k={'Case.Number': 'CASE-14825444', 'Description': 'hi rfc  my system', 'Category': 'shipment.status', 'is_fyi': 'no', 'Subject': 'PO Status'}
pd.read_json(k)