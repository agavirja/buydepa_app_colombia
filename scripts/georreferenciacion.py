import json
import requests
from formato_direccion import formato_direccion

def georreferenciacion(direccion):
    direccion     = formato_direccion(direccion)
    direccion     = f'{direccion},bogota,colombia'
    googlemapskey = 'AIzaSyAgT26vVoJnpjwmkoNaDl1Aj3NezOlSpKs'
    punto         = requests.get(f'https://maps.googleapis.com/maps/api/geocode/json?address={direccion}&key={googlemapskey}')
    response      = json.loads(punto.text)['results']
    result        = {'latitud':response[0]["geometry"]["location"]['lat'],'longitud':response[0]["geometry"]["location"]['lng'],'direccion':response[0]["formatted_address"]}
    return result