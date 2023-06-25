import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine

user     = st.secrets["user_appraisal"]
password = st.secrets["password_appraisal"]
host     = st.secrets["host_appraisal"]
schema   = st.secrets["schema_appraisal"]

def tracking(email,tipo,search=None):
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    df     =  pd.read_sql_query(f"""SELECT id as iduser,token  FROM {schema}.users WHERE email='{email}';""" , engine)
    if df.empty is False:
        dataexport = pd.DataFrame([{'date':datetime.now().strftime('%Y-%m-%d %H:%M'),'type':tipo,'search':search,'iduser':df['iduser'].iloc[0],'token':df['token'].iloc[0]}])
        dataexport.to_sql('user_tracking', engine, if_exists='append', index=False, chunksize=1)
    engine.dispose()