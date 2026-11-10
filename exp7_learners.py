from common import *
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb, lightgbm as lgb
R=10
def mk(n,r):
    if n=='RF': return RandomForestRegressor(100,min_samples_leaf=2,random_state=r,n_jobs=1)
    if n=='HGB': return HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10,random_state=r)
    if n=='XGB': return xgb.XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=4,min_child_weight=5,subsample=0.9,colsample_bytree=0.8,reg_lambda=5,n_jobs=1,random_state=r)
    if n=='LGBM': return lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=15,min_child_samples=10,subsample=0.9,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1,n_jobs=1,verbose=-1,random_state=r)
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs,gi=np.unique(g,return_inverse=True); res={}
    for n in ('RF','HGB','XGB','LGBM'):
        for sch in ('random','grouped'):
            P=np.zeros((R,len(y)))
            for r in range(R):
                spl=(KFold(5,shuffle=True,random_state=r).split(Xv) if sch=='random' else GroupKFold(5,shuffle=True,random_state=r).split(Xv,y,g))
                for tr,te in spl: P[r,te]=mk(n,r).fit(Xv[tr],y[tr]).predict(Xv[te])
            res[f'{n}_{sch}']=P
            print(tn,n,sch,round(float(np.mean([1-((y-p)**2).sum()/((y-y.mean())**2).sum() for p in P])),3),flush=True)
    np.savez(f'exp7_{tn}.npz',y=y,gi=gi,**res)
