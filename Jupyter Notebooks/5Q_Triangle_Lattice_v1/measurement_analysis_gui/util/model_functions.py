import numpy as np

def lorentzian_fit(x, x0, a, b, c):
    return a/(b+np.power((x-x0), 2))+c

def transmon_model_fit(x, x0, a, b, c, d):
    return np.sqrt(a*np.sqrt(np.power(np.cos(b*(x-x0)),2) + (d**2)*np.power(np.sin(b*(x-x0)),2))) - c