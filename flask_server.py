# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 20:00:18 2018

@author: venkatesh.s49
"""

import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
from flask import json
#import score_option1
from metadata import MetaData
from flask import Response
app = Flask(__name__)

@app.route('/api/model/classify', methods=['POST'])
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
#    try:
    print(request.get_json())
    test_json = request.get_json()
    
    k=MetaData(test_json)
    int_res=k.getData()
    print('------------------------------')
    print(int_res)
#    df=pd.DataFrame()
#    df=df.append(test_json,ignore_index=True,)

#    predicted_dict=score_option1.trigger(df)
    
#    response=json.dumps(int_res)
#    print('response:'+str(type(response)))

#    a={'textId': 3446, 'classes': [{'values': [{'name': 'expedite.request', 'confidence': 0.988201428983004}, {'name': 'quote.status', 'confidence': 0.007472280264189597}, {'name': 'internal.team.request.to.order.status.team', 'confidence': 0.0021826520995282357}, {'name': 'shipment.status', 'confidence': 0.0016337290045511444}, {'name': 'for.your.information.cases', 'confidence': 0.0005099096487268356}], 'name': 'category'}], 'modelId': 100}
    return jsonify(int_res)
#    return (response)
#    return Response(int_res, content_type='text/json; charset=utf-8')

if __name__ == '__main__':
   app.run(host='127.0.0.1',port=5000,debug = True)
