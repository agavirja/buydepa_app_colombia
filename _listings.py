import streamlit as st
import pandas as pd
import folium
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from folium.plugins import Draw
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from shapely.geometry import Polygon,mapping,shape
from price_parser import Price


#st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

# http://leaflet.github.io/Leaflet.draw/docs/leaflet-draw-latest.html#drawoptions

@st.experimental_memo
def getdatamarketcoddir(tabla,consulta):  
    user     = st.secrets["user_market"]
    password = st.secrets["password_market"]
    host     = st.secrets["host_market"]
    schema   = st.secrets["schema_market"]
    
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    st.session_state.data = pd.read_sql_query(f"""
    SELECT code, direccion, imagen_principal, areaconstruida, valorventa, valorarriendo, habitaciones, banos, garajes, estrato, antiguedad, latitud, longitud
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY direccion, areaconstruida, habitaciones, banos, garajes, {st.session_state.vardep}
                   ORDER BY code) AS row_num
        FROM market.{tabla}
        {consulta}
    ) AS filtered_market
    WHERE filtered_market.row_num = 1
    LIMIT 200;
    """, engine)
    if st.session_state.data.empty is False:
        st.session_state.latitud    = st.session_state.data['latitud'].median()
        st.session_state.longitud   = st.session_state.data['longitud'].median()
        st.session_state.zoom_start = 15

    st.experimental_rerun()


def style_function(feature):
    return {
        'fillColor': '#ffaf00',
        'color': 'blue',
        'weight': 0, 
        #'dashArray': '5, 5'
    }    
    
def onchangemin():
    if st.session_state.habitacionmin>st.session_state.habitacionmax: st.session_state.habitacionmin = st.session_state.habitacionmax
    if st.session_state.banosmin>st.session_state.banosmax:           st.session_state.banosmin      = st.session_state.banosmax
    if st.session_state.garajesmin>st.session_state.garajesmax:       st.session_state.garajesmin    = st.session_state.garajesmax
    if st.session_state.estratomin>st.session_state.estratomax:       st.session_state.estratomin    = st.session_state.estratomax

def onchangemax():
    if st.session_state.habitacionmax<st.session_state.habitacionmin: st.session_state.habitacionmax = st.session_state.habitacionmin
    if st.session_state.banosmax<st.session_state.banosmin:           st.session_state.banosmax      = st.session_state.banosmin
    if st.session_state.garajesmax<st.session_state.garajesmin:       st.session_state.garajesmax    = st.session_state.garajesmin
    if st.session_state.estratomax<st.session_state.estratomin:       st.session_state.estratomax    = st.session_state.estratomin

def onchang_tiponegocio():
    if st.session_state.tiponegocio.lower()=='venta':
        st.session_state.vardep = 'valorventa'
    if st.session_state.tiponegocio.lower()=='arriendo':
        st.session_state.vardep = 'valorarriendo'

def change_preciomin():
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    if valuemin>valuemax: st.session_state.preciomin = st.session_state.preciomax
    #if st.session_state.preciomin>st.session_state.preciomax: st.session_state.preciomin = st.session_state.preciomax
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    try:    st.session_state.preciomin = f'${valuemin:,.0f}'
    except: st.session_state.preciomin = '$0'
    
def change_preciomax():
    valuemin = Price.fromstring(st.session_state.preciomin).amount_float
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    if valuemax<valuemin: st.session_state.preciomax = st.session_state.preciomin
    #if st.session_state.preciomax<st.session_state.preciomin: st.session_state.preciomax = st.session_state.preciomin
    valuemax = Price.fromstring(st.session_state.preciomax).amount_float
    try:    st.session_state.preciomax = f'${valuemax:,.0f}'
    except: st.session_state.preciomax = '$0'
     
def onfilter():
    if st.session_state.filterdata=='Menor precio':
        st.session_state.data = st.session_state.data.sort_values(by=[st.session_state.vardep],ascending=True)
    if st.session_state.filterdata=='Mayor precio':
        st.session_state.data = st.session_state.data.sort_values(by=[st.session_state.vardep],ascending=False)
    if st.session_state.filterdata=='Menor área':
        st.session_state.data = st.session_state.data.sort_values(by=['areaconstruida'],ascending=True)
    if st.session_state.filterdata=='Mayor área':
        st.session_state.data = st.session_state.data.sort_values(by=['areaconstruida'],ascending=False)
    if st.session_state.filterdata=='Menor habitaciones':
        st.session_state.data = st.session_state.data.sort_values(by=['habitaciones'],ascending=True)
    if st.session_state.filterdata=='Mayor habitaciones':
        st.session_state.data = st.session_state.data.sort_values(by=['habitaciones'],ascending=False)

def main():
    formato = {'access':False,
               'tiponegocio':'Venta', 
               'vardep':'valorventa',
               'preciomin':'$100,000,000',
               'preciomax':'$2,500,000,000',
               'tipoinmueble':'Apartamento',
               'polygonfilter':None,
               'zoom_start':12,
               'latitud':4.652652, 
               'longitud':-74.077899,
               'habitacionmin':1,
               'habitacionmax':6,
               'banosmin':1,
               'banosmax':6,
               'garajesmin':0,
               'garajesmax':5,
               'estratomin':2,
               'estratomax':6,
               'data':pd.DataFrame()}

    for key,value in formato.items():
        if key not in st.session_state: 
            st.session_state[key] = value
    
    col1, col2, col3 = st.columns([4,1,1])
    with col1:
    
        m = folium.Map(location=[st.session_state.latitud, st.session_state.longitud], zoom_start=st.session_state.zoom_start,tiles="cartodbpositron")
     
        if st.session_state.polygonfilter is not None:
            img_style = '''
                    <style>               
                        .property-image{
                          flex: 1;
                        }
                        img{
                            width:200px;
                            height:120px;
                            object-fit: cover;
                            margin-bottom: 2px; 
                        }
                    </style>
                    '''
            for i, inmueble in st.session_state.data.iterrows():
                if isinstance(inmueble['imagen_principal'], str) and len(inmueble['imagen_principal'])>20: imagen_principal =  inmueble['imagen_principal']
                else: imagen_principal = "https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png"
                url_export   = f"buydepa-app-colombia.streamlit.app/Ficha?code={inmueble['code']}&tiponegocio={st.session_state.tiponegocio}&tipoinmueble={st.session_state.tipoinmueble}" 
                string_popup = f'''
                <!DOCTYPE html>
                <html>
                  <head>
                    {img_style}
                  </head>
                  <body>
                      <div>
                      <a href="{url_export}" target="_blank">
                      <div class="property-image">
                          <img src="{imagen_principal}"  alt="property image" onerror="this.src='https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png';">
                      </div>
                      </a>
                      <b> Direccion: {inmueble['direccion']}</b><br>
                      <b> Precio: ${inmueble[st.session_state.vardep]:,.0f}</b><br>
                      <b> Área: {inmueble['areaconstruida']}</b><br>
                      <b> Habitaciones: {int(inmueble['habitaciones'])}</b><br>
                      <b> Baños: {int(inmueble['banos'])}</b><br>
                      <b> Garajes: {int(inmueble['garajes'])}</b><br>
                      </div>
                  </body>
                </html>
                '''
                folium.Marker(location=[inmueble["latitud"], inmueble["longitud"]], popup=string_popup).add_to(m)
    
        draw = Draw(
                    draw_options={"polyline": False,"marker": False,"circlemarker":False,"rectangle":False,"circle":False},
                    edit_options={"poly": {"allowIntersection": False}}
                    )
        draw.add_to(m)
        
        if st.session_state.polygonfilter is not None:
            geojson_data                = mapping(st.session_state.polygonfilter)
            polygon_shape               = shape(geojson_data)
            centroid                    = polygon_shape.centroid
            st.session_state.latitud    = centroid.y
            st.session_state.longitud   = centroid.x
            st.session_state.zoom_start = 16
            folium.GeoJson(geojson_data, style_function=style_function).add_to(m)
    
        st_map = st_folium(m,width=1200,height=700)
        
        polygonType = ''
        if 'all_drawings' in st_map and st_map['all_drawings'] is not None:
            if st_map['all_drawings']!=[]:
                if 'geometry' in st_map['all_drawings'][0] and 'type' in st_map['all_drawings'][0]['geometry']:
                    polygonType = st_map['all_drawings'][0]['geometry']['type']
            
        if 'point' in polygonType.lower():
            coordenadas   = st_map['last_circle_polygon']['coordinates']
            st.session_state.polygonfilter = Polygon(coordenadas[0])
            st.experimental_rerun()
            
        if 'polygon' in polygonType.lower():
            coordenadas   = st_map['all_drawings'][0]['geometry']['coordinates']
            st.session_state.polygonfilter = Polygon(coordenadas[0])
            st.experimental_rerun()
    
    with col2:
        tipoinmueblefilter = st.selectbox('Tipo inmueble',key='tipoinmueble',options=['Apartamento','Casa'])
        #st.session_state.tipoinmueble = copy.deepcopy(tipoinmueblefilter)
        
    with col3:
        tiponegociofilter  = st.selectbox('Negocio',key='tiponegocio',options=['Venta','Arriendo'],on_change=onchang_tiponegocio)
        #st.session_state.tiponegocio = copy.deepcopy(tiponegociofilter)
        
    # Filtro por Precio
    with col2:
        if st.session_state.tiponegocio.lower()=='venta':
            preciomin = st.text_input('Precio mínimo',value='$100,000,000',key='preciomin',on_change=change_preciomin)
        if st.session_state.tiponegocio.lower()=='arriendo':
            preciomin = st.text_input('Precio mínimo',value='$1,000,000',key='preciomin',on_change=change_preciomin)
        preciomin = Price.fromstring(preciomin).amount_float
        
    with col3:
        if st.session_state.tiponegocio.lower()=='venta':
            preciomax = st.text_input('Precio máximo',value='$2,500,000,000',key='preciomax',on_change=change_preciomax)
        if st.session_state.tiponegocio.lower()=='arriendo':
            preciomax = st.text_input('Precio máximo',value='$20,000,000',key='preciomax',on_change=change_preciomax)
        preciomax = Price.fromstring(preciomax).amount_float
        
    # Filtro por area     
    with col2:
        areamin = st.number_input('Área construida mínima',value=25)
    with col3:
        areamax = st.number_input('Área construida máxima',value=500)
    
    # Filtro por antiguedad    
    with col2:
        antiguedamin = st.number_input('Antigueda mínima',value=0)
    with col3:
        antiguedamax = st.number_input('Antigueda máxima',value=50)
    
    # Filtro por habitaciones
    with col2:
        habitacionmin = st.selectbox('Habitaciones mínimas',options=[1,2,3,4,5,6],key='habitacionmin',on_change=onchangemin)
    with col3:
        habitacionmax = st.selectbox('Habitaciones máximas',options=[1,2,3,4,5,6],key='habitacionmax',on_change=onchangemax)
    
    # Filtro por banos
    with col2:
        banosmin = st.selectbox('Baños mínimos',options=[1,2,3,4,5,6],key='banosmin',on_change=onchangemin)
    with col3:
        banosmax = st.selectbox('Baños máximos',options=[1,2,3,4,5,6],key='banosmax',on_change=onchangemax)
    
    # Filtro por garajes
    with col2:
        garajesmin = st.selectbox('Garajes mínimos',options=[0,1,2,3,4,5],key='garajesmin',on_change=onchangemin)
    with col3:
        garajesmax = st.selectbox('Garajes máximos',options=[0,1,2,3,4,5],key='garajesmax',on_change=onchangemax)
    
    # Filtro por estrato
    with col2:
        estratomin = st.selectbox('Estrato mínimo',options=[1,2,3,4,5,6],key='estratomin',on_change=onchangemin)
    with col3:
        estratomax = st.selectbox('Estrato máximo',options=[1,2,3,4,5,6],key='estratomax',on_change=onchangemax)
            
    
    inputvar = [
        {'name':'habitaciones','min':st.session_state.habitacionmin,'max':st.session_state.habitacionmax},
        {'name':'banos','min':st.session_state.banosmin,'max':st.session_state.banosmax},
        {'name':'garajes','min':st.session_state.garajesmin,'max':st.session_state.garajesmax},
        {'name':'estrato','min':st.session_state.estratomin,'max':st.session_state.estratomax},
        {'name':st.session_state.vardep,'min':preciomin,'max':preciomax},
        {'name':'areaconstruida','min': areamin,'max':areamax},
        {'name':'antiguedad'  ,'min':antiguedamin,'max':antiguedamax},
        ]
    
    # st.session_state.polygonfilter.wkt
    # st.session_state.polygonfilter.to_wkt()
    
    with col2:
        if st.button('Buscar Inmuebles'):
            consulta = ""
            for i in inputvar:
                consultapaso = ""
                if 'min' in i and i['min'] is not None:
                    consultapaso += f' AND `{i["name"]}`>={i["min"]}'
                if 'max' in i and i['max'] is not None:
                    consultapaso += f' AND `{i["name"]}`<={i["max"]}'
                if consultapaso!='':
                    consultapaso = consultapaso.strip().strip('AND').strip()
                    consulta    += f' AND ({consultapaso})'
                
            if st.session_state.polygonfilter is not None:
                consulta += f' AND ST_CONTAINS(ST_GEOMFROMTEXT("{st.session_state.polygonfilter}"),geometry)'
                st.session_state.zoom_start = 15
    
            if consulta!='':
                consulta = consulta.strip().strip('AND').strip()
                consulta = f'WHERE {consulta}'
                    
            tabla = f'data_colombia_bogota_{st.session_state.tiponegocio.lower()}_{st.session_state.tipoinmueble.lower()}_market'
            getdatamarketcoddir(tabla,consulta)
            #st.write(inputvar)
     
    with col3:            
        if st.button('Resetear Busqueda'):
            for key,value in formato.items():
                if key in st.session_state:
                    del st.session_state[key]
                    st.session_state[key] = value
            st.experimental_rerun()
            
    components.html(
        """
    <script>
    const elements = window.parent.document.querySelectorAll('.stButton button')
    elements[0].style.backgroundColor = 'lightblue' 
    elements[1].style.backgroundColor = 'lightwite'
    </script>
    """,    
        height=0,
        width=0,
    )
    
    #-------------------------------------------------------------------------#
    # Inmuebles
    if st.session_state.data.empty is False:
        st.write('---') 
        st.markdown('<div style="background-color: #f2f2f2; border: 1px solid #fff; padding: 0px; margin-bottom: 10px;"><h1 style="margin: 0; font-size: 18px; text-align: center; color: #3A5AFF;">Listado de inmuebles</h1></div>', unsafe_allow_html=True)
        col1,col2 = st.columns([1,3])
        with col1:
            filtro = st.selectbox('Filtro por:', options=['Sin filtrar','Menor precio','Mayor precio','Menor área','Mayor área','Menor habitaciones','Mayor habitaciones'],key='filterdata',on_change=onfilter)
    
        css_format = """
            <style>
              .property-card-left {
                width: 100%;
                 /* height: 1600px; */
                 /* overflow-y: scroll; */
                text-align: center;
                display: inline-block;
                margin: 0px auto;
              }
        
              .property-block {
                width:32%;
                background-color: white;
                border: 1px solid gray;
                box-shadow: 2px 2px 2px gray;
                padding: 3px;
                margin-bottom: 10px; 
          	    display: inline-block;
          	    float: left;
                margin-right: 10px; 
              }
        
              .property {
                border: 1px solid gray;
                box-shadow: 2px 2px 2px gray;
                padding: 10px;
                margin-bottom: 10px;
              }
              
              .property-image{
                flex: 1;
              }
              .property-info{
                flex: 1;
              }
              
              .price-info {
                font-family: 'Comic Sans MS', cursive;
                font-size: 24px;
                margin-bottom: 1px;
              }
         
              .admon-info {
                font-family: 'Comic Sans MS', cursive;
                font-size: 12px;
                margin-bottom: 5px;
              }
              
              .caracteristicas-info {
                font-size: 16px;
                margin-bottom: 2px;
              }
        
              img{
                max-width: 100%;
                width: 100%;
                height:250px;
                object-fit: cover;
                margin-bottom: 10px; 
              }
            </style>
        """
        
        imagenes = ''
        for i, inmueble in st.session_state.data.iterrows():
        
            if isinstance(inmueble['imagen_principal'], str) and len(inmueble['imagen_principal'])>20: imagen_principal =  inmueble['imagen_principal']
            else: imagen_principal = "https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png"
            caracteristicas = f'<strong>{inmueble["areaconstruida"]}</strong> mt<sup>2</sup> | <strong>{int(inmueble["habitaciones"])}</strong> hab | <strong>{int(inmueble["banos"])}</strong> baños | <strong>{int(inmueble["garajes"])}</strong> pq'
            url_export      = f"buydepa-app-colombia.streamlit.app/Ficha?code={inmueble['code']}&tiponegocio={st.session_state.tiponegocio}&tipoinmueble={st.session_state.tipoinmueble}"
            
            if pd.isnull(inmueble['direccion']): direccionlabel = '<p class="caracteristicas-info">&nbsp</p>'
            else: direccionlabel = f'''<p class="caracteristicas-info">Dirección: {inmueble['direccion'][0:35]}</p>'''
            
            imagenes += f'''
                  <div class="property-block">
                    <a href="{url_export}" target="_blank">
                    <div class="property-image">
                      <img src="{imagen_principal}" alt="property image" onerror="this.src='https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png';">
                    </div>
                    </a>
                    <p class="price-info">${inmueble[st.session_state.vardep]:,.0f}</h3>
                    {direccionlabel}
                    <p class="caracteristicas-info">{caracteristicas}</p>
                  </div>
                  '''
        texto = f"""
            <!DOCTYPE html>
            <html>
              <head>
              {css_format}
              </head>
              <body>
            <div class="property-card-left">
            {imagenes}
            </div>
              </body>
            </html>
            """
        texto = BeautifulSoup(texto, 'html.parser')
        st.markdown(texto, unsafe_allow_html=True)