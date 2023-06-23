import re

def coddir(x):
    result = x
    try: result = prefijo(x) + getnewdir(x)
    except:pass
    return result

def getdirformat(x):
    # x    = 'carrera 19a # 103A - 62'
    result = ''
    x      = x.lower()
    x      = re.sub(r'[^0-9a-zA-Z]',' ', x).split(' ')
    for u in range(len(x)):
        i=x[u]
        try: i = i.replace(' ','').strip().lower()
        except: pass
        try:
            float(re.sub(r'[^0-9]',' ', i))
            result = result +'+'+i
        except:
            if i!='': result = result + i
        try:
            if len(re.sub(r'[^+]','',result))>=3:
                try:
                    if 'sur'  in x[u+1]:  result= result + 'sur'
                    if 'este' in x[u+1]:  result= result + 'este'
                except: pass
                break
        except: pass
    return result

def getnewdir(x):
    result = None
    try:
        x      = getdirformat(x).split('+')[1:]
        result = ''
        for i in x:
            result = result + '+' + re.sub(r'[^0-9]','', i)+''.join([''.join(sorted(re.sub(r'[^a-zA-Z]','', i)))])
    except: pass
    if result=='': result = None
    return result

def prefijo(x):
    result = None
    m      = re.search("\d", x).start()
    x      = x[:m].strip()
    prefijos = {'D':{'d','diagonal','dg', 'diag', 'dg.', 'diag.', 'dig'},
                'T':{'t','transv', 'tranv', 'tv', 'tr', 'tv.', 'tr.', 'tranv.', 'transv.', 'transversal', 'tranversal'},
                'C':{'c','avenida calle','avenida cll','avenida cl','calle', 'cl', 'cll', 'cl.', 'cll.', 'ac', 'a calle', 'av calle', 'av cll', 'a cll'},
                'AK':{'avenida carrera','avenida cr','avenida kr','ak', 'av cr', 'av carrera', 'av cra'},
                'K':{'k','carrera', 'cr', 'cra', 'cr.', 'cra.', 'kr', 'kr.', 'kra.', 'kra'},
                'A':{'av','avenida'}}
    for key, values in prefijos.items():
        if x.lower() in values:
            result = key
            break
    return result