import numpy as np, pandas as pd, json, time
from sklearn.model_selection import GroupKFold, KFold
from sklearn.ensemble import ExtraTreesRegressor
import os
D=os.environ.get('RAC_DATA_DIR','.')+'/'
def load(f,t):
    d=pd.read_excel(D+f); d['src']=d.source_reference.ffill(); d=d[d.outlier_flag==0].reset_index(drop=True)
    X=d[['ρ-RCA (kg/m3)','CV-RCA (%)','W-RCA (%)','Cement (kg/m3)','FA (kg/m3)','GGBS (kg/m3)','Water (kg/m3)','Sand (kg/m3)','CA (kg/m3)','RAR (%)','SP (kg/m3)']].copy()
    X.columns=['rho','cv','wabs','cem','fa','ggbs','water','sand','ca','rar','sp']
    return X,d[t].values.astype(float),d.src.values
def eng(X):
    E=X.copy(); b=X.cem+X.fa+X.ggbs
    E['wb']=X.water/b; E['binder']=b; E['scm']=(X.fa+X.ggbs)/b; E['sp_b']=X.sp/b
    E['ew']=X.water+X.ca*X.rar/100*X.wabs/100; E['ewb']=E.ew/b; E['sandr']=X.sand/(X.sand+X.ca); E['paste']=b+X.water
    return E
TG={'CS':('compressive_strength_database.xlsx','CS (MPa)'),'SL':('slump_database.xlsx','SL (mm)')}
def ET(n=100,seed=0): return ExtraTreesRegressor(n_estimators=n,random_state=seed,n_jobs=1)
