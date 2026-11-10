"""Run on your own computer:  pip install xgboost lightgbm shap scikit-learn pandas openpyxl
Same data, features, outer folds and nested source-grouped tuning as tune.py, for XGBoost and LightGBM,
plus exact TreeSHAP on held-out sources. Put tune.py in the same folder and edit DATA_DIR below.
NOTE: this script was NOT executed in the environment where the manuscript was prepared."""
import numpy as np, json, shap, xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error
DATA_DIR='.'   # folder containing the two .xlsx files
src=open('tune.py').read().split("NITER=8")[0].replace("/mnt/project/",DATA_DIR+"/")
exec(src)
SPACE={'XGB':dict(n_estimators=[200,400],learning_rate=[0.02,0.05,0.1],max_depth=[3,4,6],min_child_weight=[1,5,10],subsample=[0.7,0.9,1.0],colsample_bytree=[0.6,0.8,1.0],reg_lambda=[1,5,10]),
       'LGBM':dict(n_estimators=[200,400],learning_rate=[0.02,0.05,0.1],num_leaves=[8,15,31],min_child_samples=[5,10,20,40],subsample=[0.7,0.9,1.0],colsample_bytree=[0.6,0.8,1.0],reg_lambda=[0,1,10])}
def make(n,p): return xgb.XGBRegressor(random_state=0,n_jobs=1,subsample_freq=1,**p) if n=='XGB' else lgb.LGBMRegressor(random_state=0,n_jobs=1,verbose=-1,subsample_freq=1,**p)
res={}
for tname,(f,t) in {'CS':('compressive_strength_database.xlsx','CS (MPa)'),'SL':('slump_database.xlsx','SL (mm)')}.items():
    X,y,g=load(f,t); outer=list(GroupKFold(5,shuffle=True,random_state=0).split(X,y,g))
    for n in ['XGB','LGBM']:
        pt=np.zeros(len(y)); sv=[]
        for k,(tr,te) in enumerate(outer):
            rng=np.random.RandomState(100+k); cands=[{a:v[rng.randint(len(v))] for a,v in SPACE[n].items()} for _ in range(20)]
            inner=list(GroupKFold(3,shuffle=True,random_state=k).split(X.iloc[tr],y[tr],g[tr])); sc=[]
            for p in cands:
                pr=np.zeros(len(tr))
                for a,b in inner: pr[b]=make(n,p).fit(X.iloc[tr].iloc[a],y[tr][a]).predict(X.iloc[tr].iloc[b])
                sc.append(mean_squared_error(y[tr],pr))
            m=make(n,cands[int(np.argmin(sc))]).fit(X.iloc[tr],y[tr]); pt[te]=m.predict(X.iloc[te])
            sv.append(np.abs(shap.TreeExplainer(m).shap_values(X.iloc[te])).mean(0))
        res[f'{tname}|{n}']=dict(r2=r2_score(y,pt),rmse=mean_squared_error(y,pt)**.5,mean_abs_shap=dict(zip(X.columns,np.mean(sv,0).round(3))))
        print(tname,n,res[f'{tname}|{n}'],flush=True)
json.dump(res,open('xgb_lgbm_shap_results.json','w'),default=float)
