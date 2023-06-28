import streamlit as st
import re
import pandas as pd
import numpy as np
import json
import requests
import folium
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from shapely.geometry import Polygon,mapping
from shapely import wkt
from price_parser import Price
from urllib.parse import quote_plus
from fuzzywuzzy import fuzz
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode

from scripts.coddir import coddir
from scripts.html_scripts import table2,table3,boxkpi,boxnumbermoney
from scripts.user_tracking import tracking

#st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

def getfromjson(x,varreplace):
    try: 
        x              = json.loads(x)
        selected_items = [(i, item) for i, item in enumerate(x) if item['value'] == varreplace]
        return x[selected_items[0][0]+1]['value']
    except: return None
    
def replacenull(data,varnull,varreplace):
    idd = data[varnull].isnull()
    if sum(idd)>0:
        data.loc[idd,varnull] = data.loc[idd,'documento_json'].apply(lambda x: getfromjson(x,varreplace))
    return data
   
def asignar_coordenadas(dataset):
    latitud  = None
    longitud = None
    if not dataset.empty and 'latitud' in dataset and 'longitud' in dataset:
        dataset = dataset[(dataset['latitud'].notnull()) & (dataset['longitud'].notnull())]
        if isinstance(dataset['latitud'].iloc[0], float) and isinstance(dataset['longitud'].iloc[0], float):
            latitud  = dataset['latitud'].iloc[0]
            longitud = dataset['longitud'].iloc[0]
    return latitud, longitud

def obtener_coordenadas(*datasets):
    for dataset in datasets:
        latitud, longitud = asignar_coordenadas(dataset)
        if latitud is not None and longitud is not None:
            return latitud, longitud
    return None, None

def orderbytype(data):
    order = 1
    data['order'] = None
    for i in ['AP','CA','CS','OF']:
        idd = data['direccion'].str.lower().str.contains(i.lower())
        if sum(idd)>0:
            data.loc[idd,'order'] = order
        order += 1
    data = data.sort_values(by='order',ascending=True)
    return data.groupby('docid').agg({'direccion':'first'}).reset_index()
    
def best_match(x,datacompare):
    datacompare['ratio'] = datacompare['direccion'].apply(lambda w: fuzz.token_sort_ratio(x, w))
    datacompare          = datacompare[datacompare['ratio']>95]
    if datacompare.empty is False:
        return datacompare['areaconstruida'].iloc[0]
    else: return None

def phoneposition(x,pos):
    try: return x[pos]
    except: return None
    
def str2num(x):
    try: return int(float(x))
    except: return x
         
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

@st.experimental_memo
def circle_polygon(metros,lat,lng):
    grados   = np.arange(-180, 190, 10)
    Clat     = ((metros/1000.0)/6371.0)*180/np.pi
    Clng     = Clat/np.cos(lat*np.pi/180.0)
    theta    = np.pi*grados/180.0
    longitud = lng + Clng*np.cos(theta)
    latitud  = lat + Clat*np.sin(theta)
    return Polygon([[x, y] for x,y in zip(longitud,latitud)])

def style_function(feature):
    return {
        'fillColor': '#1ba3f2',
        'color': 'blue',
        'weight': 0, 
        #'dashArray': '5, 5'
    }    
    
def dir2comp(x,pos):
    try: 
        x = re.sub(r'(\d+)', r',\1', x)
        componentes = x.split(',')
        componentes = [c.strip() for c in componentes if c.strip() != '']
        return re.sub(r'(\d+)([A-Za-z]+)', r'\1 \2', componentes[pos]) 
    except: return None
    
@st.experimental_memo
def conjuntos_direcciones():
    user     = st.secrets["user_bigdata"]
    password = st.secrets["password_bigdata"]
    host     = st.secrets["host_bigdata"]
    schema   = st.secrets["schema_bigdata"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')    
    data     = pd.read_sql_query("""SELECT coddir,direccion,nombre_conjunto FROM bigdata.data_bogota_conjuntos """ , engine)
    
    idd = data['nombre_conjunto'].astype(str).str.contains('"')
    if sum(idd)>0:
        data.loc[idd,'nombre_conjunto'] = data.loc[idd,'nombre_conjunto'].apply(lambda x: x.strip('"'))
    data['via'] = data['direccion'].apply(lambda x: dir2comp(x,0))
    v           = data['via'].value_counts().reset_index()
    v           = v[v['count']>50]
    idd         = data['via'].isin(v['via'])
    if sum(~idd)>0:
        data.loc[~idd,'via'] = None
    data['via'] = data['via'].replace(['CL', 'KR', 'TV', 'AK', 'AC', 'DG'],['Calle', 'Carrera', 'Transversal', 'Avenida Carrera', 'Avenida Calle', 'Diagonal'])
    
    data['comp1'] = data['direccion'].apply(lambda x: dir2comp(x,1))
    data['comp2'] = data['direccion'].apply(lambda x: dir2comp(x,2))
    data['comp3'] = data['direccion'].apply(lambda x: dir2comp(x,3))
    idd = (data['via'].notnull()) & (data['comp1'].notnull()) & (data['comp2'].notnull()) & (data['comp3'].notnull())
    data['new_dir'] = None
    data.loc[idd,'new_dir'] = data.loc[idd,'via']+' '+data.loc[idd,'comp1']+' '+data.loc[idd,'comp2']+' '+data.loc[idd,'comp3'] 
    return data

@st.cache(allow_output_mutation=True)
def getlatlng(direccion):
    api_key  = "AIzaSyAgT26vVoJnpjwmkoNaDl1Aj3NezOlSpKs"
    latitud  = None
    longitud = None
    direccion_codificada = quote_plus(direccion)
    url      = f"https://maps.googleapis.com/maps/api/geocode/json?address={direccion_codificada}&key={api_key}"
    response = requests.get(url)
    data     = response.json()

    if data['status'] == 'OK':
        latitud = data['results'][0]['geometry']['location']['lat']
        longitud = data['results'][0]['geometry']['location']['lng']
    return latitud, longitud

@st.cache(allow_output_mutation=True)
def getdatanivel1(fcoddir):
    user     = st.secrets["user_bigdata"]
    password = st.secrets["password_bigdata"]
    host     = st.secrets["host_bigdata"]
    schema   = st.secrets["schema_bigdata"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')

    dataconjunto = pd.read_sql_query(f"""SELECT * FROM bigdata.data_bogota_conjuntos WHERE coddir='{fcoddir}';""" , engine)
    datapredios  = pd.read_sql_query(f"""SELECT barmanpre as lotcodigo, estrato, piso, predirecc,preaconst,prechip,precedcata FROM bigdata.data_bogota_catastro WHERE coddir='{fcoddir}';""" , engine)
    datalote     = pd.DataFrame()
    if datapredios.empty is False:
        lotcodigo    = datapredios['lotcodigo'].iloc[0]
        datalote     = pd.read_sql_query(f"""SELECT ST_AsText(geometry) as wkt FROM bigdata.data_bogota_lotes WHERE lotcodigo='{lotcodigo}';""" , engine)
    engine.dispose()
    return dataconjunto,datapredios,datalote

@st.cache(allow_output_mutation=True)
def getdatanivel2(fcoddir,datapredios):
    user     = st.secrets["user_bigdata"]
    password = st.secrets["password_bigdata"]
    host     = st.secrets["host_bigdata"]
    schema   = st.secrets["schema_bigdata"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')

    datadir      = pd.read_sql_query(f"""SELECT direccion, matricula FROM bigdata.snr_matricula_geometry WHERE coddir='{fcoddir}';""" , engine)
    dataprocesos = pd.DataFrame()
    if datadir.empty is False:
        listamat = ', '.join(f'"{i}"' for i in datadir['matricula'].unique().tolist())
        dataids  = pd.read_sql_query(f"""SELECT docid, value as matricula FROM bigdata.snr_data_matricula WHERE value IN ({listamat})""", engine)
        dataids  = dataids.merge(datadir,on='matricula',how='left',validate='m:1')
        if dataids.empty is False:
            docidlist    = ', '.join(f'{i}' for i in dataids['docid'].unique().tolist())  
            dataprocesos = pd.read_sql_query(f"""SELECT docid,nombre,tarifa,cuantia FROM bigdata.snr_tabla_procesos WHERE docid IN ({docidlist}) AND codigo IN ('125') AND cuantia>0""", engine)
            if dataprocesos.empty is False:
                docidlist = ', '.join(f'{i}' for i in dataprocesos['docid'].unique().tolist())  
                dataids   = dataids[dataids['docid'].isin(dataprocesos['docid'])]
            datacompleta = pd.read_sql_query(f"""SELECT docid,tipo_documento_publico,numero_documento_publico,fecha_documento_publico,oficina,entidad,documento_json FROM bigdata.snr_data_completa WHERE docid IN ({docidlist})""", engine)
            if datacompleta.empty is False:
                datacompleta = replacenull(datacompleta,'fecha_documento_publico','Fecha:')
                if 'documento_json' in datacompleta:
                    datacompleta.drop(columns=['documento_json'],inplace=True)
            if dataprocesos.empty is False:
                dataprocesos = dataprocesos.merge(datacompleta,on='docid',how='left',validate='m:1')
                dataids      = orderbytype(dataids)
                dataprocesos = dataprocesos.merge(dataids,on='docid',how='left',validate='m:1')
                dataprocesos = dataprocesos.sort_values(by='fecha_documento_publico',ascending=False)
                
                try:
                    if datapredios.empty is False:
                        datamerge    = datapredios[['predirecc','preaconst']]
                        datamerge.rename(columns={'predirecc':'direccion','preaconst':'areaconstruida'},inplace=True)
                        datamerge    = datamerge.sort_values(by='areaconstruida',ascending=False)
                        datamerge    = datamerge.drop_duplicates(subset='direccion',keep='first')
                        dataprocesos = dataprocesos.merge(datamerge,on='direccion',how='left',validate='m:1')
                        idd          = dataprocesos['areaconstruida'].isnull()
                        if sum(idd)>0:
                            dataprocesos.loc[idd,'areaconstruida'] = dataprocesos.loc[idd,'direccion'].apply(lambda x: best_match(x,datamerge.copy()))
                        variables    = ['docid','direccion','areaconstruida','fecha_documento_publico','nombre','tarifa', 'cuantia', 'tipo_documento_publico', 'numero_documento_publico','oficina', 'entidad']
                        variables    = [x for x in variables if x in dataprocesos]
                        dataprocesos = dataprocesos[variables]
                except: pass
    engine.dispose()
    return dataprocesos
    
@st.cache(allow_output_mutation=True)
def getdatanivel3(fcoddir):
    user     = st.secrets["user_market"]
    password = st.secrets["password_market"]
    host     = st.secrets["host_market"]
    schema   = st.secrets["schema_market"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    
    one_year_ago = datetime.now() - timedelta(days=545)
    one_year_ago = one_year_ago.strftime('%Y-%m-%d')
    
    #datamarketventa    = pd.read_sql_query(f"""SELECT direccion,descripcion,url,tipoinmueble,fecha_inicial,areaconstruida,valormt2,rango,habitaciones,banos,garajes,img1,valorventa FROM appraisal.colombia_venta_apartamento_market WHERE coddir='{fcoddir}';""" , engine)
    #datamarketarriendo = pd.read_sql_query(f"""SELECT direccion,descripcion,url,tipoinmueble,fecha_inicial,areaconstruida,valormt2,rango,habitaciones,banos,garajes,img1,valorarriendo FROM appraisal.colombia_arriendo_apartamento_market WHERE coddir='{fcoddir}';""" , engine)
    #datagaleria        = pd.read_sql_query(f"""SELECT * FROM market.data_galeria_usados_bogota WHERE coddir='{fcoddir}';""" , engine)

    datamarketventa    = pd.read_sql_query(f"""SELECT code, direccion,descripcion,url,tipoinmueble,fecha_inicial,areaconstruida,valormt2,habitaciones,banos,garajes,imagen_principal as img1,valorventa FROM market.data_colombia_bogota_venta_apartamento_market WHERE coddir='{fcoddir}';""" , engine)
    datamarketarriendo = pd.read_sql_query(f"""SELECT code, direccion,descripcion,url,tipoinmueble,fecha_inicial,areaconstruida,valormt2,habitaciones,banos,garajes,imagen_principal as img1,valorarriendo FROM market.data_colombia_bogota_arriendo_apartamento_market WHERE coddir='{fcoddir}';""" , engine)
    datagaleria        = pd.read_sql_query(f"""SELECT fecha_inicial,tipo_cliente,tipoinmueble,tiponegocio,telefono1,telefono2,telefono3,telefono4 FROM market.data_galeria_usados_bogota WHERE coddir='{fcoddir}'  AND fecha_inicial>='{one_year_ago}' AND available=1""" , engine)
    
    if datagaleria.empty is False:
        datagaleria['listaphone'] = datagaleria[['telefono1','telefono2','telefono3','telefono4']].apply(lambda x: [w for w in x.to_list() if w is not None] ,axis=1)
        for i in range(1,5):
            datagaleria[f'telefono{i}'] = datagaleria['listaphone'].apply(lambda x: phoneposition(x,i-1))
        del datagaleria['listaphone']
        
    datamarketventa['rango']    = pd.cut(datamarketventa['areaconstruida'],[0,30,40,60,100,150,200,300,np.inf],labels=['0-30','30-40','40-60','60-100','100-150','150-200','200-300','300-max'])
    datamarketarriendo['rango'] = pd.cut(datamarketarriendo['areaconstruida'],[0,30,40,60,100,150,200,300,np.inf],labels=['0-30','30-40','40-60','60-100','100-150','150-200','200-300','300-max'])
    engine.dispose()
    return datamarketventa,datamarketarriendo,datagaleria

@st.cache(allow_output_mutation=True)
def getdatanivel4(fcoddir):
    user     = st.secrets["user_colombia"]
    password = st.secrets["password_colombia"]
    host     = st.secrets["host_colombia"]
    schema   = st.secrets["schema_colombia"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    datarecorrido = pd.read_sql_query(f"""SELECT fecha_recorrido as fecha_inicial, tipo_negocio as tiponegocio, telefono1, telefono2, telefono3 FROM colombia.app_recorredor_stock_ventanas WHERE coddir='{fcoddir}';""" , engine)
    if datarecorrido.empty is False:
        for i in ['telefono1', 'telefono2', 'telefono3']:
            idd = datarecorrido[i]==''
            if sum(idd)>0:
                datarecorrido.loc[idd,i] = None
        datarecorrido['listaphone'] = datarecorrido[['telefono1','telefono2','telefono3']].apply(lambda x: [w for w in x.to_list() if w is not None] ,axis=1)
        for i in range(1,4):
            datarecorrido[f'telefono{i}'] = datarecorrido['listaphone'].apply(lambda x: phoneposition(x,i-1))
        del datarecorrido['listaphone']
        datarecorrido = datarecorrido.dropna(subset=['telefono1'])
    engine.dispose()
    return datarecorrido

@st.cache(allow_output_mutation=True)
def getdatanivel5(latitud,longitud):
    user     = st.secrets["user_appraisal"]
    password = st.secrets["password_appraisal"]
    host     = st.secrets["host_appraisal"]
    schema   = st.secrets["schema_appraisal"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    
    databarrio            = pd.DataFrame()
    barriopricing         = pd.DataFrame()
    barriocaracterizacion = pd.DataFrame()
    barriovalorizacion    = pd.DataFrame()
        
    if latitud is not None or longitud is not None:
        databarrio = pd.read_sql_query(f"""SELECT *,ST_AsText(geometry) as wkt FROM appraisal.barrios WHERE st_contains(geometry,point({longitud},{latitud}))""" , engine)
    
    if databarrio.empty is False:
        codigo             = databarrio['codigo'].iloc[0]
        tablaventa         = 'colombia_venta_apartamento_barrio'
        tablaarriendo      = 'colombia_arriendo_apartamento_barrio'
        databarrioventa    = pd.read_sql_query(f"""SELECT * FROM appraisal.{tablaventa} WHERE codigo='{codigo}'"""  , engine)
        databarrioarriendo = pd.read_sql_query(f"""SELECT * FROM appraisal.{tablaarriendo} WHERE codigo='{codigo}'""" , engine)
        
        barriopricing = pd.DataFrame()
        if databarrioventa.empty is False:
            databarrioventa['tiponegocio'] = 'Venta'
            barriopricing = pd.concat([barriopricing,databarrioventa])
        if databarrioarriendo.empty is False:
            databarrioarriendo['tiponegocio'] = 'Arriendo'
            barriopricing = pd.concat([barriopricing,databarrioarriendo])
        
        if barriopricing.empty is False:
            barriopricing['combinacion'] = None
            idd = barriopricing['tipo']=='barrio'
            if sum(idd)>0:
                barriopricing.loc[idd,'combinacion'] = ''
                
            idd = barriopricing['tipo']=='complemento'
            if sum(idd)>0:
                barriopricing.loc[idd,'combinacion'] = barriopricing.loc[idd,'habitaciones'].astype(int).astype(str)+' H + '+barriopricing.loc[idd,'banos'].astype(int).astype(str)+' B'
    
            idd = barriopricing['tipo']=='complemento_garaje'
            if sum(idd)>0:
                barriopricing.loc[idd,'combinacion'] = barriopricing.loc[idd,'habitaciones'].astype(int).astype(str)+' H + '+barriopricing.loc[idd,'banos'].astype(int).astype(str)+' B + '+barriopricing.loc[idd,'garajes'].astype(int).astype(str)+' G'
            
        tablaventa               = 'colombia_venta_apartamento_valorizacion'
        tablaarriendo            = 'colombia_arriendo_apartamento_valorizacion'
        datavalorizacionventa    = pd.read_sql_query(f"""SELECT * FROM appraisal.{tablaventa} WHERE codigo='{codigo}'"""  , engine)
        datavalorizacionarriendo = pd.read_sql_query(f"""SELECT * FROM appraisal.{tablaarriendo} WHERE codigo='{codigo}'""" , engine)
    
        barriovalorizacion = pd.DataFrame()
        if datavalorizacionventa.empty is False:
            datavalorizacionventa['tiponegocio'] = 'Venta'
            barriovalorizacion = pd.concat([barriovalorizacion,datavalorizacionventa])
        if datavalorizacionarriendo.empty is False:
            datavalorizacionarriendo['tiponegocio'] = 'Arriendo'
            barriovalorizacion = pd.concat([barriovalorizacion,datavalorizacionarriendo])
            
        if barriovalorizacion.empty is False:
            barriovalorizacion['combinacion'] = None
            idd = barriovalorizacion['tipo']=='barrio'
            if sum(idd)>0:
                barriovalorizacion.loc[idd,'combinacion'] = ''
                
            idd = barriovalorizacion['tipo']=='complemento'
            if sum(idd)>0:
                barriovalorizacion.loc[idd,'combinacion'] = barriovalorizacion.loc[idd,'habitaciones'].astype(int).astype(str)+' H + '+barriovalorizacion.loc[idd,'banos'].astype(int).astype(str)+' B'
    
            idd = barriovalorizacion['tipo']=='complemento_garaje'
            if sum(idd)>0:
                barriovalorizacion.loc[idd,'combinacion'] = barriovalorizacion.loc[idd,'habitaciones'].astype(int).astype(str)+' H + '+barriovalorizacion.loc[idd,'banos'].astype(int).astype(str)+' B + '+barriovalorizacion.loc[idd,'garajes'].astype(int).astype(str)+' G'
        
        tablaventa                  = 'colombia_venta_apartamento_caracterizacion'
        tablaarriendo               = 'colombia_arriendo_apartamento_caracterizacion'
        datacaracterizacionventa    = pd.read_sql_query(f"""SELECT variable,valor,tipo FROM appraisal.{tablaventa} WHERE codigo='{codigo}'"""  , engine)
        datacaracterizacionarriendo = pd.read_sql_query(f"""SELECT variable,valor,tipo FROM appraisal.{tablaarriendo} WHERE codigo='{codigo}'""" , engine)
        
        barriocaracterizacion = pd.DataFrame()
        if datacaracterizacionventa.empty is False:
            datacaracterizacionventa['tiponegocio'] = 'Venta'
            barriocaracterizacion = pd.concat([barriocaracterizacion,datacaracterizacionventa])
        if datacaracterizacionarriendo.empty is False:
            datacaracterizacionarriendo['tiponegocio'] = 'Arriendo'
            barriocaracterizacion = pd.concat([barriocaracterizacion,datacaracterizacionarriendo])
            
    engine.dispose()
    return databarrio,barriopricing,barriocaracterizacion,barriovalorizacion

@st.cache(allow_output_mutation=True)
def getdatadocid(docid):
    user     = st.secrets["user_bigdata"]
    password = st.secrets["password_bigdata"]
    host     = st.secrets["host_bigdata"]
    schema   = st.secrets["schema_bigdata"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')

    dataprocesos = pd.read_sql_query(f"""SELECT nombre,tarifa,cuantia FROM bigdata.snr_tabla_procesos WHERE docid='{docid}';""" , engine)
    engine.dispose()
    return dataprocesos


@st.cache(allow_output_mutation=True)
def getdatacatastro(chip):
    user     = st.secrets["user_bigdata"]
    password = st.secrets["password_bigdata"]
    host     = st.secrets["host_bigdata"]
    schema   = st.secrets["schema_bigdata"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')

    data             = pd.read_sql_query(f"""SELECT vigencia, nroIdentificacion,valorAutoavaluo,valorImpuesto,indPago,fechaPresentacion FROM bigdata.data_bogota_catastro_vigencia WHERE chip='{chip}'""" , engine)
    datamatricula    = pd.read_sql_query(f"""SELECT numeroMatriculaInmobiliaria FROM bigdata.data_bogota_catastro_predio WHERE numeroChip='{chip}'""" , engine)
    datapropietarios = pd.DataFrame()
    if datamatricula.empty is False:
        if data.empty:
            data.loc[0,'matricula'] = datamatricula['numeroMatriculaInmobiliaria'].iloc[0]
        else:
            data['matricula'] = datamatricula['numeroMatriculaInmobiliaria'].iloc[0]
    
    try: numdocument = data[data['nroIdentificacion'].notnull()]['nroIdentificacion'].iloc[0]
    except: numdocument = None
    if numdocument is not None and len(numdocument)>3:
        datapropietarios = pd.read_sql_query(f"""SELECT nroIdentificacion,tipoPropietario,matriculaMercantil,tipoDocumento,primerNombre,segundoNombre,primerApellido,segundoApellido,email,telefonos FROM bigdata.data_bogota_catastro_propietario WHERE nroIdentificacion='{numdocument}'""" , engine)
    engine.dispose()
    return data,datapropietarios

@st.cache(allow_output_mutation=True)
def getdatacomparativo(lat,lng,tiponegocio,tipoinmueble,areamin,areamax,valormin,valormax,habitaciones,banos,garajes,metros=500):
    user     = st.secrets["user_market"]
    password = st.secrets["password_market"]
    host     = st.secrets["host_market"]
    schema   = st.secrets["schema_market"]
    engine   = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    tabla    = f'data_colombia_bogota_{tiponegocio.lower()}_{tipoinmueble.lower()}_market'

    if tiponegocio.lower()=='venta':
        query = f"""SELECT direccion,url,areaconstruida,valorventa,valorarriendo,latitud,longitud,code,habitaciones,banos,garajes,imagen_principal as img1 FROM market.{tabla} WHERE (areaconstruida>={areamin} AND areaconstruida<={areamax}) AND (valorventa>={valormin} AND valorventa<={valormax}) AND habitaciones={habitaciones} AND banos={banos} AND garajes={garajes} AND ST_Distance_Sphere(geometry,POINT({lng},{lat}))<={metros}"""
    elif tiponegocio.lower()=='arriendo':
        query = f"""SELECT direccion,url,areaconstruida,valorventa,valorarriendo,latitud,longitud,code,habitaciones,banos,garajes,imagen_principal as img1 FROM market.{tabla} WHERE (areaconstruida>={areamin} AND areaconstruida<={areamax}) AND (valorarriendo>={valormin} AND valorarriendo<={valormax}) AND habitaciones={habitaciones} AND banos={banos} AND garajes={garajes} AND ST_Distance_Sphere(geometry,POINT({lng},{lat}))<={metros}"""
    data  = pd.read_sql_query(query , engine)
    engine.dispose()
    
    if data.empty is False:
        if tiponegocio.lower()=='venta':
            data['valor'] = data['valorventa'].copy()
        elif tiponegocio.lower()=='arriendo':
            data['valor'] = data['valorarriendo'].copy()
            
        data['valormt2'] = data['valor']/data['areaconstruida']

    return data
    
def change_preciomin():
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    if valuemin>valuemax: st.session_state.preciomin = st.session_state.preciomax
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    try:    st.session_state.preciomin = f'${valuemin:,.0f}'
    except: st.session_state.preciomin = '$0'
    
def change_preciomax():
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    if valuemax<valuemin: st.session_state.preciomax = st.session_state.preciomin
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    try:    st.session_state.preciomax = f'${valuemax:,.0f}'
    except: st.session_state.preciomax = '$0'


def change_ed_nombre():
    datafilter = conjuntos_direcciones()
    if st.session_state.ed_nombre!='':
        datafilter = datafilter[datafilter['nombre_conjunto']==st.session_state.ed_nombre]
        idd        = (datafilter['new_dir'].isnull()) | (datafilter['new_dir']=='')
        if sum(~idd)>0:
            st.session_state.options_ed_dir = datafilter[~idd]['new_dir'].unique()
            st.session_state.ed_dir = st.session_state.options_ed_dir[0]
            st.session_state.options_ed_dir = list(sorted(st.session_state.options_ed_dir))
            
    if st.session_state.ed_nombre=='':
        idd             = (datafilter['new_dir'].isnull()) | (datafilter['new_dir']=='')
        st.session_state.options_ed_dir = datafilter[~idd]['new_dir'].unique()
        st.session_state.options_ed_dir = ['']+list(sorted(st.session_state.options_ed_dir)) 
        st.session_state.ed_dir = ''
        
def change_ed_dir():
    datafilter                         = conjuntos_direcciones()
    idd                                = (datafilter['nombre_conjunto'].isnull()) | (datafilter['nombre_conjunto']=='')
    st.session_state.options_ed_nombre = ['']+list(sorted(datafilter[~idd]['nombre_conjunto'].unique()))
    if st.session_state.ed_dir!='':
        datafilter = datafilter[datafilter['new_dir']==st.session_state.ed_dir]
        idd        = (datafilter['nombre_conjunto'].isnull()) | (datafilter['nombre_conjunto']=='')
        if sum(~idd)>0:
            st.session_state.ed_nombre = datafilter[~idd]['nombre_conjunto'].iloc[0]
    
def main():
    formato = {
                'coddir':'',
                'preciomin':'$100,000,000',
                'preciomax':'$2,500,000,000',
                'ed_nombre':'',
                'ed_dir':'',
                'datacomparables':pd.DataFrame()
               }
    
    for key,value in formato.items():
        if key not in st.session_state: 
            st.session_state[key] = value
         
    datadirconj = conjuntos_direcciones()
    if 'options_ed_nombre' not in st.session_state:
        idd             = (datadirconj['nombre_conjunto'].isnull()) | (datadirconj['nombre_conjunto']=='')
        st.session_state.options_ed_nombre = datadirconj[~idd]['nombre_conjunto'].unique()
        st.session_state.options_ed_nombre = ['']+list(sorted(st.session_state.options_ed_nombre))
    
    if 'options_ed_dir' not in st.session_state:
        idd             = (datadirconj['new_dir'].isnull()) | (datadirconj['new_dir']=='')
        st.session_state.options_ed_dir = datadirconj[~idd]['new_dir'].unique()
        st.session_state.options_ed_dir = ['']+list(sorted(st.session_state.options_ed_dir))
    
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.selectbox('Nombre del edificio',key='ed_nombre',options=st.session_state.options_ed_nombre,on_change=change_ed_nombre)
    
    with col2:
        st.selectbox('Dirección',key='ed_dir',options=st.session_state.options_ed_dir,on_change=change_ed_dir)
        
    with col3:
        direccion = st.text_input('Dirección del edificio',value=st.session_state.ed_dir)
        
    with col4:
        st.write('')
        st.write('')
        if st.button('Buscar'):
            if direccion!='': 
                st.session_state.coddir = coddir(direccion)
                tracking(st.session_state.email,'building',st.session_state.coddir)
                #st.experimental_rerun()

        
    if st.session_state.coddir!='':
        st.write('---')
        #-------------------------------------------------------------------------#
        # DATA
        #-------------------------------------------------------------------------#
        dataconjunto,datapredios,datalote              = getdatanivel1(st.session_state.coddir)
        dataprocesos                                   = getdatanivel2(st.session_state.coddir,datapredios)
        datamarketventa,datamarketarriendo,datagaleria = getdatanivel3(st.session_state.coddir)
        datarecorrido                                  = getdatanivel4(st.session_state.coddir)
        
        latitud, longitud = obtener_coordenadas(dataconjunto, datamarketventa, datamarketarriendo)
        if latitud is None or longitud is None:
            latitud, longitud = getlatlng(direccion)
            
        databarrio,barriopricing,barriocaracterizacion,barriovalorizacion = getdatanivel5(latitud,longitud)
    
        #-------------------------------------------------------------------------#
        # DESCRIPCION Y MAPA
        #-------------------------------------------------------------------------#
        if dataconjunto.empty is False:
            col1, col2 = st.columns(2)
    
            formato = [{'name':'Localidad','value':'locnombre'},
                       {'name':'Barrio','value':'barrio'},
                       {'name':'Dirección','value':'direccion'},
                       {'name':'Nombre del edificio','value':'nombre_conjunto'},
                       {'name':'Estrato','value':'estrato'},
                       {'name':'Antiguedad','value':'vetustez_median'},
                       {'name':'# pisos','value':'maxpiso'},
                       {'name':'# unidades','value':'unidades'},
                       ]
            
            html = ""
            for i in formato:
                htmlpaso = ""
                if i['value'] in dataconjunto:
                    htmlpaso += f"""
                    <td>{i["name"]}</td>
                    <td>{dataconjunto[i['value']].iloc[0]}</td>            
                    """
                    html += f"""
                        <tr>
                        {htmlpaso}
                        </tr>
                    """
            texto = BeautifulSoup(table2(html,'Descripción'), 'html.parser')
            with col1:
                st.markdown(texto, unsafe_allow_html=True)          
            
        if datalote.empty is False:
            with col2:
                geojson_data = mapping(wkt.loads(datalote["wkt"].iloc[0]))
                poly         = wkt.loads(datalote["wkt"].iloc[0])
                m            = folium.Map(location=[poly.centroid.y, poly.centroid.x], zoom_start=18,tiles="cartodbpositron")
                folium.GeoJson(geojson_data).add_to(m)
                st_map = st_folium(m,width=800,height=450)
    
    
        #-------------------------------------------------------------------------#
        # INMUEBLES VENDIDOS
        #-------------------------------------------------------------------------#  
        if dataprocesos.empty is False:
            st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;"><b>Histórico de inmuebles vendidos</b></h1></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: 
                label       = '<label>Total transacciones<br>(últimos 4 años)</label>'
                html        = boxkpi(len(dataprocesos),label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)
                
            with col2:
                one_year_ago = datetime.now() - timedelta(days=365)
                try:
                    idd     = dataprocesos['Fecha']>=one_year_ago
                    datevarname = 'Fecha'
                except: 
                    idd = dataprocesos['fecha_documento_publico']>=one_year_ago
                    datevarname =  'fecha_documento_publico'
                if sum(idd)>0:
                    label = '<label>Trasnacciones<br>(último año)</label>'
                    html  = boxkpi(sum(idd),label)
                else:
                    yearc    = dataprocesos[datevarname].apply(lambda x: x.year)
                    maxyearc = yearc.max()
                    label    = f'<label>Trasnacciones<br>(Año {maxyearc})</label>'
                    html     = sum(yearc==maxyearc)
                    html     = boxkpi(html,label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)  
                
            with col3:
                df = dataprocesos[dataprocesos['nombre']=='COMPRAVENTA']
                if df.empty is False:
                    df['valormt2']   = df['cuantia']/df['areaconstruida']
                    valormt2building = df['valormt2'].median()
                    label       = '<label>Valor por mt2<br>(referencia del edificio)</label>'
                    html        = boxkpi(f'${valormt2building:,.0f}',label)
                    html_struct = BeautifulSoup(html, 'html.parser')
                    st.markdown(html_struct, unsafe_allow_html=True)
                
            col1, col2 = st.columns(2)
            with col1:
                datatable            = dataprocesos[['direccion', 'areaconstruida', 'fecha_documento_publico', 'cuantia']].copy()
                datatable['cuantia'] = datatable['cuantia'].apply(lambda x: f'${x:,.0f}')
                datatable.rename(columns={'direccion': 'Predio','areaconstruida':'Area construida', 'nombre': 'Tipo de proceso', 'tarifa': 'Tarifa', 'cuantia': 'Valor', 'tipo_documento_publico': 'Tipo', 'numero_documento_publico': '# documento', 'fecha_documento_publico': 'Fecha', 'oficina': 'Oficina registro', 'entidad': 'Notaria'},inplace=True)
    
                gb = GridOptionsBuilder.from_dataframe(datatable)
                gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True, resizable=True,filterable=True,sortable=True,)
                gb.configure_selection(selection_mode="single", use_checkbox=True) # "multiple"
                gb.configure_side_bar(filters_panel=False,columns_panel=False)
                gridoptions = gb.build()
                
                response_close = AgGrid(
                    datatable,
                    height=350,
                    gridOptions=gridoptions,
                    enable_enterprise_modules=False,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=True,
                    header_checkbox_selection_filtered_only=False,
                    columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW,
                    use_checkbox=True)
            
            with col2:
                if response_close['selected_rows']:
                    datapaso          = dataprocesos[dataprocesos['direccion']==response_close['selected_rows'][0]['Predio']]
                    dataprocesosdocid = getdatadocid(datapaso['docid'].iloc[0])
    
                    formato = [{'name':'Dirección','value':'direccion'},
                               {'name':'Área construida','value':'areaconstruida'},
                               {'name':'Fecha documento','value':'fecha_documento_publico'},
                               {'name':'Escritura','value':'numero_documento_publico'},
                               {'name':'Oficina registro','value':'oficina'},
                               {'name':'Notaria','value':'entidad'},
    
                               ]
                    
                    html = ""
                    for i in formato:
                        htmlpaso = ""
                        if i['value'] in datapaso:
                            htmlpaso += f"""
                            <td>{i["name"]}</td>
                            <td>{datapaso[i['value']].iloc[0]}</td>            
                            """
                            html += f"""
                                <tr>
                                {htmlpaso}
                                </tr>
                            """
                    if 'docid' in datapaso:
                        htmlpaso = f"""
                        <td>Documento</td>
                        <td><a href="https://radicacion.supernotariado.gov.co/app/static/ServletFilesViewer?docId={datapaso['docid'].iloc[0]}">link</a></td>           
                        """
                        html += f"""
                            <tr>
                            {htmlpaso}
                            </tr>
                        """
           
                    texto = BeautifulSoup(table2(html,'Descripción'), 'html.parser')
                    st.markdown(texto, unsafe_allow_html=True)                     
                    
                    if dataprocesosdocid.empty is False:
                        html = ""
                        for _,iterow in dataprocesosdocid.iterrows():
                            htmlpaso = f"""
                            <td>{iterow['nombre']}</td>
                            <td>{iterow['tarifa']}</td>
                            <td>${iterow['cuantia']:,.0f}</td>   
                            """
                            html += f"""
                                <tr>
                                {htmlpaso}
                                </tr>
                            """
                                
                        texto = BeautifulSoup(table3(html,'Proceso','%','Valor'), 'html.parser')
                        st.markdown(texto, unsafe_allow_html=True)  
                                            
        #-------------------------------------------------------------------------#
        # DESCRIPCION PREDIO
        #-------------------------------------------------------------------------#  
        if datapredios.empty is False:
            st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;"><b>Descripción de cada predio del edificio</b></h1></div>', unsafe_allow_html=True)
    
            col1, col2, col3 = st.columns([1,2,2])
            
            with col1:
                df = datapredios[['predirecc']]
                df = df.sort_values(by='predirecc',ascending=True)
                df.rename(columns={'predirecc':'Dirección','preaconst':'Area construida','prechip':'Chip','precedcata':'Cedula catastral','piso':'Piso'},inplace=True)
                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
                gb.configure_selection(selection_mode="single", use_checkbox=True) # "multiple"
                gb.configure_side_bar(filters_panel=False,columns_panel=False)
                gridoptions = gb.build()
                
                response = AgGrid(
                    df,
                    height=350,
                    gridOptions=gridoptions,
                    enable_enterprise_modules=False,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=True,
                    header_checkbox_selection_filtered_only=False,
                    use_checkbox=True)
            
            if response['selected_rows']:
                with col2:
                    datainfopredio = datapredios[datapredios['predirecc']==response['selected_rows'][0]['Dirección']]
                    chip           = datainfopredio['prechip'].iloc[0]
                    datavigencia, datapropietario = getdatacatastro(chip)
                    if datavigencia.empty is False:
                        for i in ['valorAutoavaluo','valorImpuesto','matricula']:
                            datainfopredio[i] = datavigencia[i].iloc[0]
                    
                    formato = [{'name':'Dirección','value':'predirecc','type':'str'},
                               {'name':'Área construida','value':'preaconst','type':'str'},
                               {'name':'Chip','value':'prechip','type':'str'},
                               {'name':'Matricula Inmobiliaria','value':'matricula','type':'str'},
                               {'name':'Cédula catastral','value':'precedcata','type':'str'},
                               {'name':'Avalúo catastral','value':'valorAutoavaluo','type':'money'},
                               {'name':'Impuesto predial','value':'valorImpuesto','type':'money'}
                               ]
                    
                    html = ""
                    for i in formato:
                        htmlpaso = ""
                        if i['value'] in datainfopredio:
                            if i['type']=='money':
                                htmlpaso += f"""
                                <td>{i["name"]}</td>
                                <td>${datainfopredio[i['value']].iloc[0]:,.0f}</td>            
                                """
                            else: 
                                htmlpaso += f"""
                                <td>{i["name"]}</td>
                                <td>{datainfopredio[i['value']].iloc[0]}</td>            
                                """
                            html += f"""
                                <tr>
                                {htmlpaso}
                                </tr>
                            """
    
                    texto = BeautifulSoup(table2(html,'Información del predio'), 'html.parser')
                    st.markdown(texto, unsafe_allow_html=True) 
                    
                    try:
                        conteo = 1
                        for i in json.loads(datapropietario['email'].iloc[0]):
                            datapropietario[f'email{conteo}'] = i['direccion']
                            conteo += 1
                    except: pass
                    try:
                        conteo = 1
                        for i in json.loads(datapropietario['telefonos'].iloc[0]):
                            datapropietario[f'telefono{conteo}'] = i['numero']
                            conteo += 1
                    except: pass               
                    variables = [x for x in ['email','telefonos'] if x in datapropietario]
                    if variables!=[]:
                        datapropietario.drop(columns=variables,inplace=True)
    
                    st.dataframe(datapropietario)
                    csv = convert_df(datapropietario)
                    st.download_button(
                       "Información propietario",
                       csv,
                       "info_predio.csv",
                       "text/csv",
                       key='info_predio'
                    )
                    
                    
                with col3:
                    v = datavigencia.drop_duplicates(subset=['vigencia'],keep='first')
                    v = v.groupby('vigencia').agg({'valorAutoavaluo':'first','valorImpuesto':'first'}).reset_index()
                    # Creando la figura
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=v['vigencia'], 
                        y=v['valorAutoavaluo'], 
                        name='Avaluo catastral',
                        marker_color='blue'))
                    
                    fig.update_layout(
                        xaxis_title="Vigencia",
                        yaxis_title="Avaluo catastral",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                    )
                    
                    st.plotly_chart(fig)   
                    
                    st.dataframe(v)
                    csv = convert_df(v)
                    st.download_button(
                       "Avalúo catastral",
                       csv,
                       "info_avaluo.csv",
                       "text/csv",
                       key='info_avaluo'
                    )
    
                
        #-------------------------------------------------------------------------#
        # OFERTA - VENTA
        #-------------------------------------------------------------------------#
        if datamarketventa.empty is False:
            st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">Inmuebles en oferta para <b>venta</b></h1></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,4])
            with col1: 
                label       = '<label>Total ofertas<br>(último año)</label>'
                html        = boxkpi(len(datamarketventa),label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)        
            
            with col2: 
                label       = '<label>Valor venta promedio<br>(mt2)</label>'
                html        = boxkpi(f'${datamarketventa["valormt2"].median():,.0f}',label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)    
            
            with col3:
                grupoofertasventa = datamarketventa.groupby(['rango','habitaciones','banos','garajes'])['valormt2'].median().reset_index()
                grupoofertasventa = grupoofertasventa[grupoofertasventa['valormt2'].notnull()]
                grupoofertasventa.index = range(len(grupoofertasventa))
                grupoofertasventa = grupoofertasventa.sort_values(by=['rango','habitaciones','banos','garajes'],ascending=True)
                grupoofertasventa['valormt2'] = grupoofertasventa['valormt2'].apply(lambda x: f'${x:,.0f}')
                idd = grupoofertasventa['garajes'].isnull()
                if sum(idd)>0:
                    grupoofertasventa.loc[idd,'garajes'] = ''
                grupoofertasventa.rename(columns={'rango':'Rango de area','habitaciones':'# Habitaciones','banos':'# Baños','garajes':'# Garajes','valormt2':'Valor por mt2'},inplace=True)
                st.dataframe(grupoofertasventa)
    
            
            imagenes = ''
            for i, inmueble in datamarketventa.iterrows():
                if isinstance(inmueble['img1'], str) and len(inmueble['img1'])>20: imagen_principal =  inmueble['img1']
                else: imagen_principal = "https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png"
                
                try:    garajes_inmueble = f' | <strong>{int(inmueble["garajes"])}</strong> pq'
                except: garajes_inmueble = ""
                    
                propertyinfo = f'<strong>{inmueble["areaconstruida"]}</strong> mt<sup>2</sup> | <strong>{int(inmueble["habitaciones"])}</strong> hab | <strong>{int(inmueble["banos"])}</strong> baños {garajes_inmueble}'
                url_export   = f"https://buydepa-app-colombia.streamlit.app/Ficha?code={inmueble['code']}&tiponegocio=Venta&tipoinmueble=Apartamento" 
    
                if isinstance(inmueble['direccion'], str): direccion = inmueble['direccion'][0:35]
                else: direccion = '&nbsp'
                imagenes += f'''    
                  <div class="propiedad">
                    <a href="{url_export}" target="_blank">
                    <div class="imagen">
                      <img src="{imagen_principal}">
                    </div>
                    </a>
                    <div class="caracteristicas">
                      <h3>${inmueble['valorventa']:,.0f}</h3>
                      <p>{direccion}</p>
                      <p>{propertyinfo}</p>
                    </div>
                  </div>
                  '''
                
            style = """
                <style>
                  .contenedor-propiedades {
                    overflow-x: scroll;
                    white-space: nowrap;
                    margin-bottom: 40px;
                    margin-top: 30px;
                  }
                  
                  .propiedad {
                    display: inline-block;
                    vertical-align: top;
                    margin-right: 20px;
                    text-align: center;
                    width: 300px;
                  }
                  
                  .imagen {
                    height: 200px;
                    margin-bottom: 10px;
                    overflow: hidden;
                  }
                  
                  .imagen img {
                    display: block;
                    height: 100%;
                    width: 100%;
                    object-fit: cover;
                  }
                  
                  .caracteristicas {
                    background-color: #f2f2f2;
                    padding: 4px;
                    text-align: left;
                  }
                  
                  .caracteristicas h3 {
                    font-size: 18px;
                    margin-top: 0;
                  }
                  .caracteristicas p {
                    font-size: 14px;
                    margin-top: 0;
                  }
                  .caracteristicas p1 {
                    font-size: 12px;
                    text-align: left;
                    width:40%;
                    padding: 8px;
                    background-color: #294c67;
                    color: #ffffff;
                    margin-top: 0;
                  }
                  .caracteristicas p2 {
                    font-size: 12px;
                    text-align: left;
                    width:40%;
                    padding: 8px;
                    background-color: #008f39;
                    color: #ffffff;
                    margin-top: 0;
                  } 
                </style>
            """
            
            html = f"""
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                {style}
              </head>
              <body>
                <div class="contenedor-propiedades">
                {imagenes}
                </div>
              </body>
            </html>
            """
            texto = BeautifulSoup(html, 'html.parser')
            st.markdown(texto, unsafe_allow_html=True)
            
    
        #-------------------------------------------------------------------------#
        # OFERTA - ARRIENDO
        #-------------------------------------------------------------------------#
        if datamarketarriendo.empty is False:
            st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">Inmuebles en oferta para <b>arriendo</b></h1></div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,4])
            with col1: 
                label       = '<label>Total ofertas<br>(último año)</label>'
                html        = boxkpi(len(datamarketarriendo),label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)        
            
            with col2: 
                label       = '<label>Valor venta promedio<br>(mt2)</label>'
                html        = boxkpi(f'${datamarketarriendo["valormt2"].median():,.0f}',label)
                html_struct = BeautifulSoup(html, 'html.parser')
                st.markdown(html_struct, unsafe_allow_html=True)    
            
            with col3:
                grupoofertasarriendo = datamarketarriendo.groupby(['rango','habitaciones','banos','garajes'])['valormt2'].median().reset_index()
                grupoofertasarriendo = grupoofertasarriendo[grupoofertasarriendo['valormt2'].notnull()]
                grupoofertasarriendo.index = range(len(grupoofertasarriendo))
                grupoofertasarriendo = grupoofertasarriendo.sort_values(by=['rango','habitaciones','banos','garajes'],ascending=True)
                grupoofertasarriendo['valormt2'] = grupoofertasarriendo['valormt2'].apply(lambda x: f'${x:,.0f}')
                idd = grupoofertasarriendo['garajes'].isnull()
                if sum(idd)>0:
                    grupoofertasarriendo.loc[idd,'garajes'] = ''
                grupoofertasarriendo.rename(columns={'rango':'Rango de area','habitaciones':'# Habitaciones','banos':'# Baños','garajes':'# Garajes','valormt2':'Valor por mt2'},inplace=True)
                st.dataframe(grupoofertasarriendo)
    
            
            imagenes = ''
            for i, inmueble in datamarketarriendo.iterrows():
                if isinstance(inmueble['img1'], str) and len(inmueble['img1'])>20: imagen_principal =  inmueble['img1']
                else: imagen_principal = "https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png"
                
                try:    garajes_inmueble = f' | <strong>{int(inmueble["garajes"])}</strong> pq'
                except: garajes_inmueble = ""
                    
                propertyinfo = f'<strong>{inmueble["areaconstruida"]}</strong> mt<sup>2</sup> | <strong>{int(inmueble["habitaciones"])}</strong> hab | <strong>{int(inmueble["banos"])}</strong> baños {garajes_inmueble}'
                url_export   = f"https://buydepa-app-colombia.streamlit.app/Ficha?code={inmueble['code']}&tiponegocio=Arriendo&tipoinmueble=Apartamento" 
    
                if isinstance(inmueble['direccion'], str): direccion = inmueble['direccion'][0:35]
                else: direccion = '&nbsp'
                imagenes += f'''    
                  <div class="propiedad">
                    <a href="{url_export}" target="_blank">
                    <div class="imagen">
                      <img src="{imagen_principal}">
                    </div>
                    </a>
                    <div class="caracteristicas">
                      <h3>${inmueble['valorarriendo']:,.0f}</h3>
                      <p>{direccion}</p>
                      <p>{propertyinfo}</p>
                    </div>
                  </div>
                  '''
                
            style = """
                <style>
                  .contenedor-propiedades {
                    overflow-x: scroll;
                    white-space: nowrap;
                    margin-bottom: 40px;
                    margin-top: 30px;
                  }
                  
                  .propiedad {
                    display: inline-block;
                    vertical-align: top;
                    margin-right: 20px;
                    text-align: center;
                    width: 300px;
                  }
                  
                  .imagen {
                    height: 200px;
                    margin-bottom: 10px;
                    overflow: hidden;
                  }
                  
                  .imagen img {
                    display: block;
                    height: 100%;
                    width: 100%;
                    object-fit: cover;
                  }
                  
                  .caracteristicas {
                    background-color: #f2f2f2;
                    padding: 4px;
                    text-align: left;
                  }
                  
                  .caracteristicas h3 {
                    font-size: 18px;
                    margin-top: 0;
                  }
                  .caracteristicas p {
                    font-size: 14px;
                    margin-top: 0;
                  }
                  .caracteristicas p1 {
                    font-size: 12px;
                    text-align: left;
                    width:40%;
                    padding: 8px;
                    background-color: #294c67;
                    color: #ffffff;
                    margin-top: 0;
                  }
                  .caracteristicas p2 {
                    font-size: 12px;
                    text-align: left;
                    width:40%;
                    padding: 8px;
                    background-color: #008f39;
                    color: #ffffff;
                    margin-top: 0;
                  } 
                </style>
            """
            
            html = f"""
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="UTF-8">
                {style}
              </head>
              <body>
                <div class="contenedor-propiedades">
                {imagenes}
                </div>
              </body>
            </html>
            """
            texto = BeautifulSoup(html, 'html.parser')
            st.markdown(texto, unsafe_allow_html=True)
    
        #-------------------------------------------------------------------------#
        # TELEFONOS DE CONTACTO
        #-------------------------------------------------------------------------#
        col1, col2 = st.columns(2)
        with col1:
            dataphones = pd.concat([datagaleria,datarecorrido])
            for i in list(dataphones):
                idd = dataphones[i].isnull()
                if sum(idd)>0:
                    dataphones.loc[idd,i] = ''
            if dataphones.empty is False:
                dataphones.rename(columns={'fecha_inicial':'Fecha','tipo_cliente':'Tipo de aviso','tipoinmueble':'Tip ode inmueble','tiponegocio':'Tipo de negocio'},inplace=True)
                st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">Teléfonos de disponibles</h1></div>', unsafe_allow_html=True)
                st.dataframe(dataphones)
            
    
        #-------------------------------------------------------------------------#
        # REFERENCIA DE PRECIOS BARRIO
        #-------------------------------------------------------------------------#
        st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">Referencia de precios en el barrio</h1></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            sel_tiponegocio = st.selectbox('Tipo de negocio',options=['Venta','Arriendo'])
        with col2:
            opciones = list(barriopricing[barriopricing['tiponegocio']==sel_tiponegocio]['combinacion'].unique())+list(barriovalorizacion[barriovalorizacion['tiponegocio']==sel_tiponegocio]['combinacion'].unique())
            opciones = list(set(opciones))
            opciones = sorted(opciones)
            opciones.remove('')
            opciones = ['']+opciones
            sel_tipologia = st.selectbox('Tipología',options=opciones)
        
        col    = st.columns(2)
        conteo = 0
        if barriopricing.empty is False:
            idd = (barriopricing['tiponegocio']==sel_tiponegocio) 
            if sel_tipologia=='':
                idd = (idd) & (barriopricing['tipo']=='barrio')
            else:
                idd = (idd) & (barriopricing['combinacion']==sel_tipologia)
            if sum(idd)>0:
                with col[conteo]:
                    valor = barriopricing[idd]['valormt2'].iloc[0]
                    obs   = barriopricing[idd]['obs'].iloc[0]
                    label = f'<label>Precio por mt <sup>2</sup><br>{sel_tiponegocio}</label>'
                    html        = boxnumbermoney(f'${valor:,.0f}' ,f'Muestra: {obs}',label)
                    html_struct = BeautifulSoup(html, 'html.parser')
                    st.markdown(html_struct, unsafe_allow_html=True) 
                    conteo += 1
                    
        if barriovalorizacion.empty is False:
            idd = (barriovalorizacion['tiponegocio']==sel_tiponegocio) 
            if sel_tipologia=='':
                idd = (idd) & (barriovalorizacion['tipo']=='barrio')
            else:
                idd = (idd) & (barriovalorizacion['combinacion']==sel_tipologia)
            if sum(idd)>0:
                with col[conteo]:
                    valor       = barriovalorizacion[idd]['valorizacion'].iloc[0]
                    label       = f'<label>Valorización anual<br>{sel_tiponegocio}</label>' 
                    html        = boxnumbermoney("{:.1%}".format(valor),'&nbsp;',label)
                    html_struct = BeautifulSoup(html, 'html.parser')
                    st.markdown(html_struct, unsafe_allow_html=True) 
                    conteo += 1                        
        #-------------------------------------------------------------------------#
        # ESTADISTICAS
        #-------------------------------------------------------------------------#
        col    = st.columns(2)
        dfpaso = barriocaracterizacion[barriocaracterizacion['tiponegocio']==sel_tiponegocio]
        if dfpaso.empty is False:
            dfpaso['variable'] = dfpaso['variable'].apply(lambda x: str2num(x))
            formato = [{'name':'areaconstruida','label':'Área construida','order':['50 o menos mt2', '50 a 80 mt2', '80 a 100 mt2', '100 a 150 mt2','150 a 200 mt2', '200 a 300 mt2','300 o más mt2']},
                       {'name':'habitaciones','label':'Habitaciones'},
                       {'name':'banos','label':'Baños'},
                       {'name':'garajes','label':'Garajes'}]
              
            conteo = 0
            for i in formato:
                df = dfpaso[dfpaso['tipo']==i['name']]
                if df.empty is False:                
                    df = df.sort_values(by='variable',ascending=True)
                    if 'order' in i:
                        df['order'] = df['variable'].replace(i['order'],range(len(i['order'])))
                        df = df.sort_values(by='order',ascending=True)
                    if conteo % 2 == 0: pos = 0
                    else: pos = 1
                    conteo += 1
                    
                    with col[pos]:
                        st.markdown(f'<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 20px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">{i["label"]}</h1></div>', unsafe_allow_html=True)            
                        fig = px.bar(df, x='variable', y='valor')
                        fig.update_traces(textposition='outside')
                        fig.update_layout(
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            xaxis_title='',
                            yaxis_title='',
                            legend_title_text=None,
                            autosize=True,
                            #xaxis={'tickangle': -90},
                            #width=800, 
                            #height=500
                        )
                        st.plotly_chart(fig, theme="streamlit",use_container_width=True) 
            
        # FALTA SECCION PARA dataconjunto: GRAFICA DE AREA DE CONSTRUCCION Y AMENITIES