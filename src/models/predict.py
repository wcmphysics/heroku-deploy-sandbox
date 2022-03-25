from logging import getLogger
import pandas as pd
import pickle
import warnings
import os
#from mlflow.sklearn import load_model

from src.data.hdd_preprocessing import load_preprocess_testdata, preprocess_testdata
from src.features.feature_engineering import hdd_preprocessor, log_transformer

warnings.filterwarnings("ignore")
logger = getLogger(__name__)

def __get_data(df):
    #X_test = load_preprocess_testdata(   days=30, filename="ST4000DM000_history_total", 
    #                                path=os.getcwd())
    X_test = preprocess_testdata(df, days=30)
    return X_test

def __get_model():
    model_path = "models/deployment_xgb/model.pkl"
    model = pickle.load(open(model_path, 'rb'))
    #model = load_model(model_path)
    return model

def run_predict(df):
    logger.info("Loading model")
    model = __get_model()
    logger.info("Loading and preprocessing data")
    X_test = __get_data(df)
    logger.info("Feature engineering on test")
    preprocessor = hdd_preprocessor(days=30, trigger=0.05)
    X_test = preprocessor.fit_transform(X_test) # Nothing saved in the fit!
    logger.info("Prediction in progress")
    y_proba = model.predict_proba(X_test)
    return y_proba > 0.15

if __name__ == "__main__":
    import logging

    logger = logging.getLogger()
    logging.basicConfig(format="%(asctime)s: %(message)s")
    logging.getLogger("pyhive").setLevel(logging.CRITICAL)  # avoid excessive logs
    logger.setLevel(logging.INFO)

    y_pred = run_predict()
    print(y_pred.sum())
