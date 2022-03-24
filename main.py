# importing the needed libraries
import pickle
import re
from fastapi import FastAPI, File, UploadFile
#from matplotlib.pyplot import axis
import uvicorn
import pandas as pd
import numpy as np
import mlflow
import json
#import requests

#from keras.models import Sequential, load_model
from mlflow.sklearn import load_model
from src.models.predict import run_predict
 


# creating the app
app = FastAPI()

# loading the model
#with open('ann_model.bin', 'rb') as f_in:
#with open('final1/model.pkl', 'rb') as f_in:
#    model = pickle.load(f_in)

#path_load = './models/linear'
#model_from_load = load_model(path_load)
#print("model loaded")

#X_test_tmp = pd.read_csv("X_test.csv").drop(["Unnamed: 0"], axis=1)
#y_test_tmp = pd.read_csv("y_test.csv").drop(["Unnamed: 0"], axis=1)


# direct reading
df = pd.read_csv("df_heroku_test.csv")
y_pred = run_predict(df)[:,1]
print(y_pred)

"""
def predict_failure(model, X_test, threshold=0.0871712944):
    y = model.predict_proba(X_test)[:,1]
    failure = np.array([ 1 if i >= threshold else 0 for i in y])
    return {'failure' : failure, 'failure_probability' : y}

"""
# Writing a function for the landing page with a fastapi decorator            
@app.get('/')
def get_root():
	return {'message': 'Welcome to the hdd failure prediction API'}

@app.post("/receive_dataframe")
async def receive_dataframe(dataframe_as_json : str):
    df = pd.DataFrame.read_json(dataframe_as_json)
    return { "Failure" : run_predict(df)[:,1]}

"""
@app.post("/uploadfile")
async def create_upload_file(data: UploadFile = File(...) ):
    #Prints result in cmd – verification purpose
    print("The file: ", data.filename, "is uploaded")
    json_data = json.load(data.file)
    print("json transformed 1")
    #json_data = json.loads(data.file.read())
    #print("json transformed 2")
    df = pd.read_json(json_data)
    print()
    #df = pd.read_csv(data)
    #Sends server the name of the file as a response
    return {"Failure": run_predict(df)}


@app.post("/dummypath")
async def get_body(request: Request):
    return await request.json()

"""





@app.post("/receive_preprocessed_dataframe")
async def receive_preprocessed_dataframe(dataframe_as_json : str):
    df = pd.DataFrame.read_json(dataframe_as_json)
    return 0

# run the app
if __name__ == '__main__':
    uvicorn.run(app, port=9696)
