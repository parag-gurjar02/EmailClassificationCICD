# This script generates the scoring and schema files
# necessary to operationalize your model
#from azureml.api.schema.dataTypes import DataTypes
#from azureml.api.schema.sampleDefinition import SampleDefinition
#from azureml.api.realtime.services import generate_schema
#from azureml.assets import get_local_path

####################### Added By Parag Start#############################
import pandas as pd
import numpy as np
import re
import pickle # This is used for creating output file.

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')
    
####################### Added By Parag Start #############################    
    #input_dump_file_location = 'D:/Parag/Projects/AINADEL/ML_experiment1/MLExperiment1/model_pkl/email_classification_multiclass_approach1_obj.pkl'
#    input_dump_file_location = './csv/email_classification_multiclass_approach1_obj.pkl'
    input_dump_file_location = 'email_classification_multiclass_approach1_obj.pkl'
    with open(input_dump_file_location,'rb') as f:  # Python 3: open(..., 'rb')
       d1_gs_clf, d1_gs_clf_estimator, \
       d1_gs_clf_svm, d1_gs_clf_svm_estimator = pickle.load(f)
####################### Added By Parag End #############################
    # Load model using appropriate library and function
    global model
    # model = model_load_function(local_path)
    model = d1_gs_clf_svm

def run(input_df):
    
    # Predict using appropriate functions
    # prediction = model.predict(input_df)
    '''
    prediction = "%s %d" % (str(input_df), model)
    return json.dumps(str(prediction))
    '''
    #it is temporary classes variable....in future classes object needs to be pickled and retrain the model.
    classes=['expedite.request','for.your.information.cases','internal.team.request.to.order.status.team','quote.status','shipment.status']
    
#    prediction1 = model.best_estimator_.predict(input_df)
    prediction = model.predict_proba(input_df)
    print(prediction[0])
#    print(dict(zip(classes,prediction[0])))
#    predicted_df=pd.DataFrame(prediction)
    
#    json_df=predicted_df.to_dict(orient='list')
    
    return dict(zip(classes,prediction[0]))
'''
def generate_api_schema():
    import os
    print("create schema")
    sample_input = "sample data text"
    inputs = {"input_df": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))
'''
# Implement test code to run in IDE or Azure ML Workbench
    

def trigger(validation_file):
#if __name__ == '__main__':
####################### Added By Parag Start #############################   
    #validation_file = 'D:/Parag/Projects/AINADEL/ML_experiment1/MLExperiment1/csv/score_data.csv'
#    validation_file = 'text_data.csv'
    ## Reading validation file
#    validation = pd.read_csv(validation_file,encoding='cp1252') #text in column 1, classifier in column 2.
#    print('.........................')
#    print(validation_file)
    
    df=pd.DataFrame()
    test_json={}
    
    test_json['Subject']=validation_file['Subject']
    test_json['Description']=validation_file['textContent']
#    print(test_json)
    df=df.append(test_json,ignore_index=True,)
    
    validation =df  #text in column 1, classifier in column 2.
    ## Replacing spaces with Nan in subject column, dropping all rows with null subject and replacing 
    ## multiple occurance of X with space.
    validation['Subject'].replace(' ', np.nan, inplace=True)
    validation.dropna(subset=['Subject'], inplace=True)
    validation['Subject'].replace(to_replace=re.compile('[x]{2,}',flags = re.IGNORECASE),
    value=' ',inplace=True,regex=True)

    ## Replacing spaces with Nan in description column, dropping all rows with null subject and replacing 
    ## multiple occurance of X with space.
    validation['Description'].replace(' ', np.nan, inplace=True)
    validation.dropna(subset=['Description'], inplace=True)
    validation['Description'].replace(to_replace=re.compile('[x]{2,}',flags = re.IGNORECASE),
    value=' ',inplace=True,regex=True)   
    
    init()
    input1 = validation[['Subject','Description']]
    json_df=run(input1)
    return json_df
####################### Added By Parag End #############################
'''
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()
    input = "{}"
    result = run(input)
    logger.log("Result",result)
'''
    


