# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 08:53:06 2018

@author: venkatesh.s49
"""

import score_option1
from operator import itemgetter

class MetaData:
    
    
    def __init__(self,json_obj1):
        
        self.json_obj1 = json_obj1
#        self.consolidated=consolidated
        
    
    def getData(self):
        
        final_json={}
        final_json['modelId']=self.json_obj1.get('modelId')
        final_json['textId']=self.json_obj1.get('textId')
        
        classes_consolidated=[]
        classesToPredict=self.json_obj1.get('classToPredict')
        
        for class_name in classesToPredict:
            class_name_values={}
            class_name_values['name']=class_name
            class_name_values['values']=self.predict(class_name) #returns list
            
            classes_consolidated.append(class_name_values)
            
        final_json['classes']=classes_consolidated
        return final_json
    
    def category(self):
        
        pred_conf_list=[]
        predicted_scores=score_option1.trigger(self.json_obj1)
        
        for category_name in list(predicted_scores):
            temp={}
            temp['name']=category_name
            temp['confidence']=predicted_scores[category_name]
            pred_conf_list.append(temp)
            
        pred_conf_list=sorted(pred_conf_list, key=itemgetter('confidence'),reverse = True)
        
        return pred_conf_list
     
    def four(self):
        return "two"
     
    def three(self):
        return "three"
    
    
    def printing(self):
        return 'venkat'
     
     
    def predict(self,argument):
        switcher = {
            'category': self.category(),
            'printing': self.printing(),
            3: self.three,
            4: self.four
            }
    
        func = switcher.get(argument, lambda: "Invalid class")
    
        return func
     
