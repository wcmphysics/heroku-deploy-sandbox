#Fetch a list of failed HDDs grouped by model and saves the dataframe as csv file
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
# Load env file with connection details
load_dotenv("../../.env")
DB_STRING = os.getenv('DB_STRING')
# Create engine to use with pandas
db = create_engine(DB_STRING)
# Query the SQL database
query_string = """SELECT model, COUNT(model), failure FROM "2021"
                GROUP BY model, failure
                HAVING failure = '1'
                ORDER BY count DESC"""
models_fail = pd.read_sql(query_string, db)
# Save the csv file
models_fail.to_csv("../../data/interim/count_failed_by_model.csv", index=False)