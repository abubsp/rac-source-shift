import numpy as np, pandas as pd, json, time, itertools
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
rng0=np.random.RandomState(0)
def load(f,t):
    d=pd.read_excel('/mnt/project/'+f); d['src']=d.source_reference.ffill(); d=d[d.outlier_flag==0].reset_index(drop=True)
    X=d[['ρ-RCA (kg/m3)','CV-RCA (%)','W-RCA (%)','Cement (kg/m3)','FA (kg/m3)','GGBS (kg/m3)','Water (kg/m3)','Sand (kg/m3)','CA (kg/m3)','RAR (%)','SP (kg/m3)']].copy()
    X.columns=['rho','cv','wabs','cem','fa','ggbs','water','sand','ca','rar','sp']
    return X,d[t].values,d.src.values
def eng(X):
    E=X.copy(); b=X.cem+X.fa+X.ggbs
    E['wb']=X.water/b; E['binder']=b; E['scm']=(X.fa+X.ggbs)/b; E['sp_b']=X.sp/b
    E['ew']=X.water+X.ca*X.rar/100*X.wabs/100; E['ewb']=E.ew/b; E['sandr']=X.sand/(X.sand+X.ca); E['paste']=b+X.water
    return E
def mk(name,p,seed=0):
    if name=='ET': return ExtraTreesRegressor(n_estimators=200,random_state=seed,n_jobs=1,**p)
    if name=='RF': return RandomForestRegressor(n_estimators=200,random_state=seed,n_jobs=1,**p)
    if name=='HGB': return HistGradientBoostingRegressor(random_state=seed,**p)
DEF={'ET':dict(min_samples_leaf=1),'RF':dict(min_samples_leaf=2),
     'HGB':dict(max_iter=400,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10)}
SP={'ET':dict(max_features=[0.4,0.6,0.8,1.0],min_samples_leaf=[1,2,3,5],max_depth=[None,12,20]),
    'RF':dict(max_features=[0.3,0.5,0.7,1.0],min_samples_leaf=[1,2,3,5],max_depth=[None,12,20]),
    'HGB':dict(learning_rate=[0.02,0.05,0.1],max_leaf_nodes=[8,15,31],min_samples_leaf=[5,10,20,40],l2_regularization=[0,0.1,1,10],max_iter=[200,400])}
def sample(name,n,rng):
    out=[]
    for _ in range(n):
        out.append({k:v[rng.randint(len(v))] for k,v in SP[name].items()})
    return out
NITER=8
res={}
for tname,(f,t) in {'CS':('compressive_strength_database.xlsx','CS (MPa)'),'SL':('slump_database.xlsx','SL (mm)')}.items():
    X0,y,g=load(f,t); X1=eng(X0)
    outer=list(GroupKFold(5,shuffle=True,random_state=0).split(X0,y,g))
    for fs,X in [('base',X0),('eng',X1)]:
        for name in (['ET','RF','HGB'] if fs=='base' else ['ET','HGB']):
            t0=time.time(); pu=np.zeros(len(y)); pt=np.zeros(len(y)); chosen=[]
            for k,(tr,te) in enumerate(outer):
                Xtr,ytr,gtr=X.iloc[tr],y[tr],g[tr]
                pu[te]=mk(name,DEF[name]).fit(Xtr,ytr).predict(X.iloc[te])
                rng=np.random.RandomState(100+k); cands=sample(name,NITER,rng)
                inner=list(GroupKFold(3,shuffle=True,random_state=k).split(Xtr,ytr,gtr))
                sc=[]
                for p in cands:
                    pr=np.zeros(len(ytr))
                    for a,b in inner: pr[b]=mk(name,p).fit(Xtr.iloc[a],ytr[a]).predict(Xtr.iloc[b])
                    sc.append(mean_squared_error(ytr,pr))
                best=cands[int(np.argmin(sc))]; chosen.append(best)
                pt[te]=mk(name,best).fit(Xtr,ytr).predict(X.iloc[te])
            r=dict(untuned_r2=r2_score(y,pu),untuned_rmse=mean_squared_error(y,pu)**.5,
                   tuned_r2=r2_score(y,pt),tuned_rmse=mean_squared_error(y,pt)**.5,chosen=chosen,
                   fold_r2_untuned=[r2_score(y[te],pu[te]) for _,te in outer],fold_r2_tuned=[r2_score(y[te],pt[te]) for _,te in outer])
            res[f'{tname}|{fs}|{name}']=r
            print(tname,fs,name,{k:round(v,3) for k,v in r.items() if isinstance(v,float)},round(time.time()-t0),'s',flush=True)
            json.dump(res,open('/tmp/ms/tune_results.json','w'),default=str)
