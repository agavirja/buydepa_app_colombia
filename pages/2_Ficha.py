import streamlit as st
import pdfcrowd
import pandas as pd
import re
import folium
from streamlit_folium import st_folium
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

import tempfile
from datetime import datetime

st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

# https://pdfcrowd.com/pricing/api/?api=v2
# https://pdfcrowd.com/user/account/stats/

user     = st.secrets["user_market"]
password = st.secrets["password_market"]
host     = st.secrets["host_market"]
schema   = st.secrets["schema_market"]

API_KEY = st.secrets["API_KEY"]
pdfcrowduser = st.secrets["pdfcrowduser"]
pdfcrowdpass = st.secrets["pdfcrowdpass"]

@st.experimental_memo
def homogenizar_texto(texto):
    # Remover múltiples espacios en blanco
    texto = re.sub(r'\s+', ' ', texto)
    # Poner todo en minúsculas, a menos que la palabra empiece después de una puntuación
    texto = re.sub(r'(?<=[^\w\s])\w+', lambda x: x.group().lower(), texto)
    # Remover caracteres no alfanuméricos
    texto = re.sub(r'[^\w\s.,;]', '', texto)
    # Remover cuando hay codigos dentro del texto
    texto = re.sub(r'C\w+ Fincaraíz: \d+', '', texto)
    # Remover telefono
    texto = re.sub(r'\b\d{7,}\b', '', texto)
    texto = texto.replace('Código Fincaraíz',' ')
    return texto

@st.experimental_memo
def getdata(code,tiponegocio,tipoinmueble):
    tabla  = f'data_colombia_bogota_{tiponegocio.lower()}_{tipoinmueble.lower()}_market'
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{schema}')
    return pd.read_sql_query(f"""SELECT mpio_cnmbr, locnombre, scanombre, tipoinmueble, direccion, latitud, longitud, estrato,areaconstruida,habitaciones,banos,garajes,valorventa,valorarriendo,antiguedad,piso,valoradministracion,descripcion,telefono1,telefono2,telefono3,email1,inmobiliaria,url, img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15 FROM market.{tabla} WHERE code='{code}'""", engine)

# obtener los argumentos de la url
args = st.experimental_get_query_params()
if 'data_ficha' not in st.session_state: 
    st.session_state.data_ficha = pd.DataFrame()
    
if 'code' in args: 
    code = args['code'][0]
else: code = ''
if 'tiponegocio' in args:
    tiponegocio = args['tiponegocio'][0]
else: tiponegocio = ''
if 'tipoinmueble' in args:
    tipoinmueble = args['tipoinmueble'][0]
else: tipoinmueble = ''

if st.session_state.data_ficha.empty:
    
    if code=='' and tiponegocio=='' and tipoinmueble=='':
        code         = st.text_input('Código',value=code)
        tipoinmueble = st.selectbox('Tipo de inmueble',options=['Apartamento','Casa',''])
        tiponegocio  = st.selectbox('Tipo de Negocio',options=['Venta','Arriendo',''])

    if code!='' and any([x for x in ['apartamento','casa'] if x in tipoinmueble.lower()]) and  any([x for x in ['venta','arriendo'] if x in tiponegocio.lower()]):
        st.session_state.data_ficha = getdata(code,tiponegocio,tipoinmueble)
        if st.session_state.data_ficha.empty:
            st.write('Inmueble no encontrado')
        st.experimental_rerun()
else:
    data_inmueble = st.session_state.data_ficha.iloc[0]

    #-------------------------------------------------------------------------#
    # Catacteristicas del inmueble
    ciudad,localidad,barrio,tipoinmueble,direccion,latitud,longitud,estrato,areaconstruida,habitaciones,banos,garajes,precio,precio = [None,None,None,None,None,None,None,None,None,None,None,None,None,None]
    
    try: ciudad        = data_inmueble['mpio_cnmbr']
    except: pass
    try:localidad      = data_inmueble['locnombre']
    except: pass
    try:barrio         = data_inmueble['scanombre']
    except: pass
    try:tipoinmueble   = data_inmueble['tipoinmueble']
    except: pass
    try:direccion      = data_inmueble['direccion']
    except: pass
    try:latitud        = data_inmueble['latitud']
    except: pass
    try:longitud       = data_inmueble['longitud']
    except: pass
    try:estrato        = int(data_inmueble['estrato'])
    except: pass
    try:areaconstruida = int(data_inmueble['areaconstruida'])
    except: pass
    try:habitaciones   = int(data_inmueble['habitaciones'])
    except: pass
    try:banos          = int(data_inmueble['banos'])
    except: pass
    try:garajes        = int(data_inmueble['garajes'])
    except: pass
    if 'venta' in tiponegocio.lower():
        vardep = 'valorventa'
    if 'arriendo' in tiponegocio.lower():
        vardep = 'valorarriendo'
    try:
        precio = data_inmueble[vardep]
        precio = f'${precio:,.0f}'
    except: pass

    try:    antiguedad = datetime.now().year-int(data_inmueble['antiguedad'])
    except: antiguedad = ""
    try:    piso = int(data_inmueble['piso'])
    except: piso = ""
    try:
        if data_inmueble['valoradministracion'] is not None and float(data_inmueble['valoradministracion'])>0:
            valoradministracion = data_inmueble['valoradministracion']
            valoradministracion  = f'<p>Administración:  ${valoradministracion:,.0f} </p>'
        else: valoradministracion = ""
    except: valoradministracion = ""
    
    caracteristicas = f'<strong>{areaconstruida}</strong> mt<sup>2</sup> | <strong>{habitaciones}</strong> habitaciones | <strong>{banos}</strong> baños | <strong>{garajes}</strong> garajes'
    try:
        descripcion     = data_inmueble['descripcion']
        descripcion     = homogenizar_texto(descripcion)
    except: descripcion = ''
    
    #-------------------------------------------------------------------------#
    # Datos de contacto
    telefono1    = data_inmueble['telefono1']
    telefono2    = data_inmueble['telefono2']
    telefono3    = data_inmueble['telefono3']
    email        = data_inmueble['email1']
    inmobiliaria = data_inmueble['inmobiliaria']
    
    #-------------------------------------------------------------------------#
    fontsize        = 13
    fontfamily      = 'sans-serif'
    backgroundcolor = '#FAFAFA'
    
    css_format = """
      <style>
        .property-card-left {
          width: 100%;
          height: 500px; /* or max-height: 300px; */
          overflow-y: scroll; /* enable vertical scrolling for the images */
        }
        .property-card-right {
          width: 100%;
          margin-left: 10px;
        }
    
        .text-justify {
          text-align: justify;
        }
        
        .no-margin {
          margin-bottom: 1px;
        }
        
        .price-part1 {
          font-family: 'Comic Sans MS', cursive;
          font-size: 24px;
          margin-bottom: 1px;
        }
        
        .price-part2 {
          font-size: 14px;
          font-family: 'Comic Sans MS';
          margin-bottom: 1px;
        }
    
        .nota {
          font-size: 12px;
        }
      img{
        max-width: 100%;
        width: 45%;
        height:250px;
        object-fit: cover;
        margin-bottom: 10px; 
      }
      </style>
    """
    
    col1, col2 = st.columns([3,2])
    with col1:
    
        imagenes  = '<div class="property-card-images">\n'
        variables = [x for x in list(st.session_state.data_ficha) if 'img' in x]
        conteo    = 0
        for i in variables:
            if isinstance(data_inmueble[i], str) and len(data_inmueble[i])>=7:
                imagenes += f'''<img src="{data_inmueble[i]}" alt="property image" onerror="this.src='https://personal-data-bucket-online.s3.us-east-2.amazonaws.com/sin_imagen.png';">\n'''
                conteo += 1
            if conteo==2:
                imagenes += '</div>\n'
                imagenes += '<div class="property-card-images">\n'
                conteo   = 0
        imagenes = BeautifulSoup(imagenes, 'html.parser')
        
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
        
    with col2:
        
        outuput_list = {'Ciudad:':None,'Dirección:':None,'Localidad:':None,'Barrio:':None,'Tipo de inmueble:':None,'Estrato:':None,'Antiguedad:':None,'Piso:':None}
        if isinstance(ciudad, str) and len(ciudad)>=4: outuput_list.update({'Ciudad:':ciudad})
        if isinstance(direccion, str) and len(direccion)>=7: outuput_list.update({'Dirección:':direccion})
        if isinstance(localidad, str) and len(localidad)>=4: outuput_list.update({'Localidad:':localidad})
        if isinstance(barrio, str) and len(barrio)>=4: outuput_list.update({'Barrio:':barrio})
        if isinstance(tipoinmueble, str) and len(tipoinmueble)>=4: outuput_list.update({'Tipo de inmueble:':tipoinmueble})
        if isinstance(estrato, int) or isinstance(estrato, float): outuput_list.update({'Estrato:':int(estrato)})
        if isinstance(antiguedad, int) or isinstance(antiguedad, float): outuput_list.update({'Antiguedad:':int(antiguedad)})
        if isinstance(piso, int) or isinstance(piso, float): outuput_list.update({'Piso:':int(piso)})

        tabla = ""
        for i,j in outuput_list.items():
            if j is not None:
                tabla += f'''
                  <tr style="border-style: none;">
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">{i}</td>
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">{j}</td>
                  </tr>
                '''
        if tabla!="":
            tabla = f'''
            <ul>
                <table style="width:100%;">
                {tabla}
                </table>
            </ul>\n
            '''
            if isinstance(descripcion, str) and len(descripcion)>=20:
                tabla += f'''
                    <h8>Descripcion:</h8>
                    <p class="text-justify">{descripcion}</p>
                '''
            tabla = BeautifulSoup(tabla, 'html.parser')

        texto_property = f"""
        <!DOCTYPE html>
        <html>
        <head>
        {css_format}
        </head>
        <body>
        <div class="property-card-right">
                <p class="no-margin">
                  <span class="price-part1">{precio}</span> 
                  <span class="price-part2">  ({tiponegocio})</span>
                </p>
                {valoradministracion}
                <p>{caracteristicas}</p>
                <p>Código: <strong>{code}</strong></p>
                {tabla}
        </div>
        </body>
        </html>
        """
        texto_property = BeautifulSoup(texto_property, 'html.parser')
        st.markdown(texto_property, unsafe_allow_html=True)
        
    st.write('---')
    col1, col2 = st.columns([3,2])
    with col1:
        map = folium.Map(location=[latitud, longitud],zoom_start=17,tiles="cartodbpositron")
        folium.Marker(location=[latitud, longitud]).add_to(map)
        st_map = st_folium(map, width=800, height=350)
        
    with col2:
        outuput_list = {'Telefono 1:':None,'Telefono 2:':None,'Telefono 3:':None,'Email contacto:':None,'Inmobilairia:':None}
        if isinstance(telefono1, str) and len(telefono1)>=7: outuput_list.update({'Telefono 1:':telefono1})
        if (isinstance(telefono1, int) or isinstance(telefono1, float)) and len(str(telefono1))>=7: outuput_list.update({'Telefono 1:':telefono1})
        if isinstance(telefono2, str) and len(telefono2)>=7: outuput_list.update({'Telefono 2:':telefono2})
        if (isinstance(telefono2, int) or isinstance(telefono2, float)) and len(str(telefono2))>=7: outuput_list.update({'Telefono 2:':telefono2})
        if isinstance(telefono3, str) and len(telefono3)>=7: outuput_list.update({'Telefono 3:':telefono3})
        if (isinstance(telefono3, int) or isinstance(telefono3, float)) and len(str(telefono3))>=7: outuput_list.update({'Telefono 3:':telefono3})
        if isinstance(email, str) and '@' in email: outuput_list.update({'Email contacto:':email})
        if isinstance(inmobiliaria, str) and len(inmobiliaria)>=4: outuput_list.update({'Inmobilairia:':inmobiliaria})
        tabla_contacto = ""
        for i,j in outuput_list.items():
            if j is not None:
                tabla_contacto += f'''
                  <tr style="border-style: none;">
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">{i}</td>
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">{j}</td>
                  </tr>
                '''
            else:
                tabla_contacto += f'''
                  <tr style="border-style: none;">
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">{i}</td>
                    <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">sin contacto</td>
                  </tr>
                '''
        if isinstance(data_inmueble['url'], str) and len(data_inmueble['url'])>20: 
            link = f'''[Link]({data_inmueble['url']})'''
            tabla_contacto += f'''
              <tr style="border-style: none;">
                <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">Ver inmueble</td>
                <td style="border-style: none;font-family:{fontfamily};font-size:{fontsize}px;">
                    <a href="{data_inmueble['url']}">Link</a>
                </td>
              </tr>
            '''                
        tabla_contacto = f'''
        <table style="width:100%;">
        {tabla_contacto}
        </table>
        '''
        tabla_contacto = BeautifulSoup(tabla_contacto, 'html.parser')
        st.write('Datos de contacto de esta propiedad')
        st.markdown(tabla_contacto, unsafe_allow_html=True)

        nota = f"""
        <!DOCTYPE html>
        <html>
        <head>
        {css_format}
        </head>
        <body>
            <p class="nota"> <strong>Nota</strong>: La información de contacto de la propiedad no sale en la ficha que se descarga en pdf</p>
        </body>
        </html>
        """
        nota = BeautifulSoup(nota, 'html.parser')
        st.markdown(nota, unsafe_allow_html=True)
    
        #---------------------------------------------------------------------#
        if st.button('Generar PDF'):
            with st.spinner("Generando PDF"):
                css_format_export = """
                        <style>
                          .property-card {
                            display: flex;
                          }
                          .property-card-left {
                            width: 60%;
                            float: left;
                          }
                          .property-card-right {
                            width: 40%;
                            float: right;
                            margin-left: 20px;
                          }
                          .price-part1 {
                            font-family: 'Comic Sans MS', cursive;
                            font-size: 24px;
                            margin-bottom: 1px;
                          }
                          
                          .price-part2 {
                            font-size: 14px;
                            font-family: 'Comic Sans MS';
                            margin-bottom: 1px;
                          }
                          .text-justify {
                            text-align: justify;
                          }
                          .no-margin {
                            margin-bottom: 1px;
                          }
                          .property-map {
                              width: 400px;
                              height: 200px;
                          }
                          img{
                              width:45%;
                              height:180px;
                              margin-bottom: 10px; 
                          }
                        </style>
                """
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                <meta charset="UTF-8">
                    {css_format_export}
                </head>
                <body>
                    <main>
                      <div class="property-card">
                        <div class="property-card-left">
                             {imagenes}
                        </div>
                      <div class="property-card-right">
                        <p class="no-margin">
                          <span class="price-part1">{precio}</span> 
                          <span class="price-part2">  ({tiponegocio})</span>
                        </p>
                        {valoradministracion}
                        <p>{caracteristicas}</p>
                        <p>Código: <strong>{code}</strong></p>
                        {tabla}
                        <img src="https://maps.googleapis.com/maps/api/staticmap?center={latitud},{longitud}&zoom=16&size=400x200&markers=color:blue|{latitud},{longitud}&key={API_KEY}" alt="Google Map" class="property-map">
                      </div>
                    </main>
                </body>
                </html>
                """
    
                caracteres_especiales = {
                    "á": "&aacute;",
                    "é": "&eacute;",
                    "í": "&iacute;",
                    "ó": "&oacute;",
                    "ú": "&uacute;",
                    "ñ": "&ntilde;",
                    "Á": "&Aacute;",
                    "É": "&Eacute;",
                    "Í": "&Iacute;",
                    "Ó": "&Oacute;",
                    "Ú": "&Uacute;",
                    "Ñ": "&Ntilde;",
                }
                for caracter, codigo in caracteres_especiales.items():
                    html = re.sub(caracter, codigo, html)

                html              = BeautifulSoup(html, 'html.parser')
                fd, temp_path     = tempfile.mkstemp(suffix=".html")
                wd, pdf_temp_path = tempfile.mkstemp(suffix=".pdf")       
                
                client = pdfcrowd.HtmlToPdfClient(pdfcrowduser,pdfcrowdpass)
                client.convertStringToFile(html, pdf_temp_path)

                with open(pdf_temp_path, "rb") as pdf_file:
                    PDFbyte = pdf_file.read()
                
                st.download_button(label="Descargar Ficha",
                                    data=PDFbyte,
                                    file_name=f"ficha-codigo-{code}.pdf",
                                    mime='application/octet-stream')
