## author: Junmei
## Date:2022-11-09
import numpy as np
from scipy import stats
import os



def rmse_metric(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=1))
## Cite: LIN L I. A Concordance Correlation-Coefficient To Evaluate Reproducibility [J]. Biometrics, 1989, 45(1): 255-68.
def ccc_original_version(x, y , ci = "z-transform", conf_level = 0.95):
    yb = y.mean()
    sy2 = y.var() 
    sd1 = y.std()

    xb = x.mean()
    sx2 = x.var() 
    sd2 = x.std()

    r =  np.diag(np.corrcoef(x, y, rowvar=True),k=1)
    sl = r * sd1 / sd2

    sxy = r * np.sqrt(sx2 * sy2)
    p = 2 * sxy / (sx2 + sy2 + (yb - xb)**2)
    return p
def ccc_modified_version(x, y ):
    yb = y.mean()
    sy2 = y.var() 
    sd1 = y.std()

    xb = x.mean()
    sx2 = x.var() 
    sd2 = x.std()

    r =  np.diag(np.corrcoef(x, y, rowvar=True),k=1)
    sl = r * sd1 / sd2
    
    cal_mse = ((y - x) ** 2).mean()
    
    sxy = r * np.sqrt(sx2 * sy2)
    p = 2 * sxy / (sx2 + sy2 + cal_mse)
    return p




