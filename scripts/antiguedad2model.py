def antiguedad2model(x):
    if x<=1:   result = 'menora1ano'
    elif x<=8: result = '1a8anos'
    elif x<=15:result = '9a15anos'
    elif x<=30:result = '16a30anos'
    else: result = 'masde30anos'
    return result