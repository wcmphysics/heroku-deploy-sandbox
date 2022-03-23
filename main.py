# importing the needed libraries
import pickle
import re
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
 
# creating the app
app = FastAPI()

# loading the model
with open('ann_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

def predict_failure(model, threshold=0.0871712944):
    prob = model.predict_proba(X_train_log_scaled)[:,1]
    failure = np.array([ 1 if i >= th else 0 for i in y])
    return {'failure' : failure, 'failure_probability' : prob}

# Writing a function for the landing page with a fastapi decorator            
@app.get('/')
def get_root():
	return {'message': 'Welcome to the hdd failure prediction API'}

@app.post("/receive_dataframe")
def receive_dataframe(dataframe_as_json : str):
    df = pd.DataFrame.read_json(dataframe_as_json)


"""


"""





# function for the classification
def classify_message(model, message):
    message = preprocessor(message)
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])
    return {'label': label, 'spam_probability': spam_prob[0][1]}



# Writing a function for the prediction with a fastapi decorator that fetches the message  
@app.get('/failure_prediction/{message}')
async def detect_spam_path(message: str):
	return classify_message(model, message)

# run the app
if __name__ == '__main__':
    uvicorn.run(app, port=9696)
