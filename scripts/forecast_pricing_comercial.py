import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import  Point,Polygon
from pricingforecast import pricingforecast
from antiguedad2model import antiguedad2model
from pricing_ponderador import pricing_ponderador

@st.cache(allow_output_mutation=True)
def getdatamarketcoddir(filename,fcoddir):
    data = pd.read_pickle(filename,compression='gzip')
    data = data[data['coddir']==fcoddir]
    return data

@st.cache(allow_output_mutation=True)
def getdatabarrio(filename,scacodigo):
    data = pd.read_pickle(filename,compression='gzip')
    data = data[(data['scacodigo']==scacodigo) & (data['tipo']=='barrio')]
    return data

@st.cache(allow_output_mutation=True)
def getdatabarriocomplemento(filename,scacodigo,inputvar):
    data = pd.read_pickle(filename,compression='gzip')
    idd  = (data['scacodigo']==scacodigo) & (data['obs']>5) & (data['tipo']=='complemento')
    if 'habitaciones' in inputvar and inputvar['habitaciones']>0:
        idd = (idd) & (data['habitaciones']==inputvar['habitaciones'])
    if 'banos' in inputvar and inputvar['banos']>0:
        idd = (idd) & (data['banos']==inputvar['banos'])
    if 'garajes' in inputvar and inputvar['garajes']>0:
        idd = (idd) & (data['garajes']==inputvar['garajes'])
    data = data[idd]
    return data

@st.cache(allow_output_mutation=True)
def getdatamarketsimilar(filename,inputvar):
    data = pd.read_pickle(filename,compression='gzip')
    idd  = True
    if 'areaconstruida' in inputvar and inputvar['areaconstruida']>0:
        areamin = inputvar['areaconstruida']*0.85
        areamax = inputvar['areaconstruida']*1.15
        idd     = (idd) & (data['areaconstruida']>=areamin)  & (data['areaconstruida']<=areamax)
    if 'habitaciones' in inputvar and inputvar['habitaciones']>0:
        idd     = (idd) & (data['habitaciones']==inputvar['habitaciones'])
    if 'banos' in inputvar and inputvar['banos']>0:
        idd     = (idd) & (data['banos']==inputvar['banos'])
    if 'garajes' in inputvar and inputvar['garajes']>0:
        idd     = (idd) & (data['garajes']==inputvar['garajes'])
    if 'estrato' in inputvar and inputvar['estrato']>0:
        idd     = (idd) & (data['estrato']==inputvar['estrato'])
    data = data[idd]
    return data

@st.cache(allow_output_mutation=True)
def getpolygon(metros,lat,lng):
    grados   = np.arange(-180, 190, 10)
    Clat     = ((metros/1000.0)/6371.0)*180/np.pi
    Clng     = Clat/np.cos(lat*np.pi/180.0)
    theta    = np.pi*grados/180.0
    longitud = lng + Clng*np.cos(theta)
    latitud  = lat + Clat*np.sin(theta)
    return Polygon([[x, y] for x,y in zip(longitud,latitud)])
        
# VERSION ADAPDATA A STREAMLIT
def forecast_pricing_comercial(inputvar):
    precioforecast      = None
    forecastlist        = {}
    
    databarrio          = pd.DataFrame()
    databarriocomp      = pd.DataFrame()
    datasimilares       = pd.DataFrame()
    dataconjunto        = pd.DataFrame()
    dataexportsimilares = pd.DataFrame()
    
    scacodigo      = None
    tiponegocio    = 'Venta'
    tipoinmueble   = 'Apartamento'
    antiguedad     = 0
    areaconstruida = 0
    habitaciones   = 0
    banos          = 0
    garajes        = 0
    estrato        = 0
    fcoddir        = ''
    latitud        = 0
    longitud       = 0
    metros         = 300
    
    if 'scacodigo'   in inputvar:  scacodigo    = inputvar['scacodigo']
    if 'tiponegocio' in inputvar:  tiponegocio  = inputvar['tiponegocio']
    if 'tipoinmueble' in inputvar: tipoinmueble = inputvar['tipoinmueble']
    if 'areaconstruida' in inputvar: areaconstruida = inputvar['areaconstruida']
    if 'habitaciones'   in inputvar: habitaciones   = inputvar['habitaciones']
    if 'banos'    in inputvar:  banos   = inputvar['banos']
    if 'garajes'  in inputvar:  garajes = inputvar['garajes']
    if 'coddir'  in inputvar:   fcoddir = inputvar['coddir']
    if 'estrato'  in inputvar:  estrato = inputvar['estrato']
    if 'latitud'  in inputvar:  latitud = inputvar['latitud']
    if 'longitud' in inputvar: longitud = inputvar['longitud']
    if 'metros'   in inputvar:   metros = inputvar['metros']
    if 'anos_antiguedad' in inputvar: antiguedad = inputvar['anos_antiguedad']
    
    if scacodigo is not None and scacodigo!='':
        if 'venta' in tiponegocio.lower():
            filename_barrio = r'D:\Dropbox\Empresa\Buydepa\COLOMBIA\DESARROLLO\proyecto-opensource-buydepa\data\data_barrio_venta_bogota'
            filename_oferta = r'D:\Dropbox\Empresa\Buydepa\COLOMBIA\DESARROLLO\proyecto-opensource-buydepa\data\data_market_venta_bogota'            
            #filename_barrio = 'data/data_barrio_venta_bogota'
            #filename_oferta = 'data/data_market_venta_bogota'

        if 'arriendo' in tiponegocio.lower():
            filename_barrio = r'D:\Dropbox\Empresa\Buydepa\COLOMBIA\DESARROLLO\proyecto-opensource-buydepa\data\data_barrio_arriendo_bogota'
            filename_oferta = r'D:\Dropbox\Empresa\Buydepa\COLOMBIA\DESARROLLO\proyecto-opensource-buydepa\data\data_market_arriendo_bogota'
            #filename_barrio = 'data/data_barrio_arriendo_bogota'
            #filename_oferta = 'data/data_market_arriendo_bogota'

        forecastlist = {'forecast_barrio':None,
                        'forecast_barrio_complemento':None,
                        'forecast_edificio_similiar':None,
                        'forecast_edificio':None,
                        'forecast_zona':None,
                        'forecast_model':None}
        
        #-------------------------------------------------------------------------#
        # Forecast: Valor por mt2 del barrio
        databarrio = getdatabarrio(filename_barrio,scacodigo)
        if databarrio.empty is False:
            if areaconstruida>0:
                precioforecast = databarrio['valormt2'].iloc[0]*areaconstruida
                forecastlist['forecast_barrio'] = precioforecast
    
        #-------------------------------------------------------------------------#
        # Forecast: Valor por mt2 del barrio, con mismas habitaciones, baños y garajes
        tipovar        = {'habitaciones':habitaciones,'banos':banos,'garajes':garajes}
        databarriocomp = getdatabarriocomplemento(filename_barrio,scacodigo,tipovar)
        if databarriocomp.empty is False:
            precioforecast   = databarriocomp['valormt2'].iloc[0]*areaconstruida
            forecastlist['forecast_barrio_complemento'] = precioforecast
    
        #-------------------------------------------------------------------------#
        # Forecast: datos del mismo conjunto
        dataconjunto = getdatamarketcoddir(filename_oferta,fcoddir)
        if dataconjunto.empty is False:
            areamin = areaconstruida*0.85
            areamax = areaconstruida*1.15
            idd     = ((dataconjunto['areaconstruida']>=areamin) & (dataconjunto['areaconstruida']<=areamax)) & (dataconjunto['habitaciones']==habitaciones) & (dataconjunto['garajes']==garajes)
            if sum(idd)>3:
                precioforecast   = dataconjunto[idd]['valormt2'].median()*areaconstruida
                forecastlist['forecast_edificio_similiar'] = precioforecast
            if len(dataconjunto)>5:
                precioforecast = dataconjunto['valormt2'].median()*areaconstruida
                forecastlist['forecast_edificio'] = precioforecast
    
        #-------------------------------------------------------------------------#
        # Forecast: datos de inmuebles similares misma zona
        inputvar                  = {'areaconstruida':areaconstruida,'habitaciones':habitaciones,'banos':banos,'estrato':estrato,'garajes':garajes}
        datasimilares             = getdatamarketsimilar(filename_oferta,inputvar)
        poly                      = getpolygon(metros,latitud,longitud)
        datasimilares['geometry'] = datasimilares.apply(lambda x: Point(x['longitud'],x['latitud']),axis=1)
        datasimilares             = gpd.GeoDataFrame(datasimilares, geometry='geometry')
        idd                       = datasimilares['geometry'].apply(lambda x: poly.contains(x))
        dataexportsimilares       = datasimilares[idd]
        if sum(idd)>(5*metros/100):
            precioforecast = dataexportsimilares[idd]['valormt2'].median()*areaconstruida
            forecastlist['forecast_zona'] = precioforecast
    
        #-------------------------------------------------------------------------#
        # Forecast: Modelo de redes neuronales
        if 'venta' in tiponegocio.lower():    modeltipo = 'sell'
        if 'arriendo' in tiponegocio.lower(): modeltipo = 'rent'
        tiempodeconstruido = antiguedad2model(antiguedad)
        forecastinputs     = {'mpio_ccdgo':'11001','scacodigo':scacodigo,'tipoinmueble':tipoinmueble,'tiponegocio':modeltipo,'areaconstruida':areaconstruida,'habitaciones':habitaciones,'banos':banos,'garajes':garajes,'estrato':estrato,'tiempodeconstruido':tiempodeconstruido}
        resultadoforecast              = pricingforecast(forecastinputs)
        forecastlist['forecast_model'] = resultadoforecast["valorestimado"]
    
    if forecastlist!={}:
        precioforecast = pricing_ponderador(forecastlist)
    return precioforecast