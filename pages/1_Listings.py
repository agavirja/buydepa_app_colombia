import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

from _listings import main

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

user     = st.secrets["user_appraisal"]
password = st.secrets["password_appraisal"]
host     = st.secrets["host_appraisal"]
schema   = st.secrets["schema_appraisal"]

@st.cache(allow_output_mutation=True)
def validate_token(token):
    if token=='':
        return False
    else:
        engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
        df       =  pd.read_sql_query(f"""SELECT * FROM {schema}.users WHERE token='{token}';""" , engine)
        engine.dispose()
        if df.empty:
            return False
        else:
            st.session_state.access = True
            for i in ['email','nombre','telefono','logo','token']:
                st.session_state[i] = df[i].iloc[0] 
            return True
        
args = st.experimental_get_query_params()

if 'access' not in st.session_state: 
    st.session_state.access = False

if 'token' not in st.session_state: 
    st.session_state.token = ''
    
if st.session_state.token=='':
    if 'token' in args: 
        st.session_state.token = args['token'][0]
        st.experimental_rerun()

if st.session_state.token!='':
    validate_token(st.session_state.token)
    
if st.session_state.access or validate_token(st.session_state.token):
    main()
    
else:
    st.error("Inicia sesión o registrate")