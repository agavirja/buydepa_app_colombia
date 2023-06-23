import pandas as pd

def pricing_ponderador(x):
    resultforecast = 0
    inputvar       = {}
    for key,value in x.items():
        try:
            if value>0:
                inputvar.update({key:value})
        except: pass
    
    if inputvar!={}:
        listainputs = list(inputvar)
        lista       = options()
        for i in lista:
            if len(listainputs)==len(i['lista']):
                suma = 0
                for j in listainputs:
                    if j in i['lista']: 
                        suma += 1
                if suma==len(listainputs):
                    listaponderacion = {}
                    for s in range(len(i['lista'])):
                        listaponderacion.update({i['lista'][s]:i['ponderacion'][s]})
                    for s in listainputs:
                        resultforecast = resultforecast + (inputvar[s]*listaponderacion[s])
                    break
                
    if resultforecast==0 and inputvar!={}:
        listapricings = []
        for key,value in inputvar.items():
            listapricings.append(value)
        w = pd.DataFrame(listapricings)
        w.columns = ['price']
        resultforecast = w['price'].median()
    
    return resultforecast
    
def options():
    options = [
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.5,0.15,0.15,0.1,0.06,0.04]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.4,0.4,0.1,0.05,0.05]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.5,0.2,0.1,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.5,0.15,0.15,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.5,0.15,0.15,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.5,0.15,0.15,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.5,0.15,0.15,0.1,0.1]},
        {'lista':['forecast_model', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.15,0.15,0.1]},
        {'lista':['forecast_edificio', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.15,0.15,0.1]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.4,0.4,0.1,0.1]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.4,0.4,0.1,0.1]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.4,0.4,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.6,0.2,0.1,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_barrio'] ,'ponderacion':[0.6,0.15,0.15,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_barrio_complemento'] ,'ponderacion':[0.6,0.15,0.15,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model', 'forecast_zona'] ,'ponderacion':[0.6,0.15,0.15,0.1]},
        {'lista':['forecast_zona', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.3,0.1]},
        {'lista':['forecast_model', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.3,0.1]},
        {'lista':['forecast_model', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.6,0.3,0.1]},
        {'lista':['forecast_model', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.6,0.3,0.1]},
        {'lista':['forecast_edificio', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.6,0.25,0.15]},
        {'lista':['forecast_edificio', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.6,0.25,0.15]},
        {'lista':['forecast_edificio', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.6,0.25,0.15]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_barrio'] ,'ponderacion':[0.5,0.4,0.1]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_barrio_complemento'] ,'ponderacion':[0.5,0.4,0.1]},
        {'lista':['forecast_edificio', 'forecast_model', 'forecast_zona'] ,'ponderacion':[0.5,0.4,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_barrio'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_barrio_complemento'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model', 'forecast_zona'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_barrio'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_barrio_complemento'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_zona'] ,'ponderacion':[0.7,0.2,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio', 'forecast_model'] ,'ponderacion':[0.6,0.2,0.2]},
        {'lista':['forecast_barrio_complemento', 'forecast_barrio'] ,'ponderacion':[0.7,0.3]},
        {'lista':['forecast_zona', 'forecast_barrio'] ,'ponderacion':[0.8,0.2]},
        {'lista':['forecast_zona', 'forecast_barrio_complemento'] ,'ponderacion':[0.7,0.3]},
        {'lista':['forecast_model', 'forecast_barrio'] ,'ponderacion':[0.9,0.1]},
        {'lista':['forecast_model', 'forecast_barrio_complemento'] ,'ponderacion':[0.8,0.2]},
        {'lista':['forecast_model', 'forecast_zona'] ,'ponderacion':[0.7,0.3]},
        {'lista':['forecast_edificio', 'forecast_barrio'] ,'ponderacion':[0.95,0.05]},
        {'lista':['forecast_edificio', 'forecast_barrio_complemento'] ,'ponderacion':[0.9,0.1]},
        {'lista':['forecast_edificio', 'forecast_zona'] ,'ponderacion':[0.8,0.2]},
        {'lista':['forecast_edificio', 'forecast_model'] ,'ponderacion':[0.7,0.3]},
        {'lista':['forecast_edificio_similiar', 'forecast_barrio'] ,'ponderacion':[0.95,0.05]},
        {'lista':['forecast_edificio_similiar', 'forecast_barrio_complemento'] ,'ponderacion':[0.92,0.08]},
        {'lista':['forecast_edificio_similiar', 'forecast_zona'] ,'ponderacion':[0.9,0.1]},
        {'lista':['forecast_edificio_similiar', 'forecast_model'] ,'ponderacion':[0.8,0.2]},
        {'lista':['forecast_edificio_similiar', 'forecast_edificio'] ,'ponderacion':[0.8,0.2]},
        {'lista':['forecast_barrio'] ,'ponderacion':[1]},
        {'lista':['forecast_barrio_complemento'] ,'ponderacion':[1]},
        {'lista':['forecast_zona'] ,'ponderacion':[1]},
        {'lista':['forecast_model'] ,'ponderacion':[1]},
        {'lista':['forecast_edificio'] ,'ponderacion':[1]},
        {'lista':['forecast_edificio_similiar'] ,'ponderacion':[1]}
        ]
    return options