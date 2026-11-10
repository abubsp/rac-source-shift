# Monte-Carlo (permutation-sampling) Shapley values for tuned-free extra-trees under source-grouped CV
import numpy as np, pandas as pd, json
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import ExtraTreesRegressor
import sys; sys.argv=['x']
exec(open('/tmp/ms/tune.py').read().split("NITER=8")[0])   # reuse load/eng/mk definitions
NPERM=60; NEXPL=60
out={}
for tname,(f,t) in {'CS':('compressive_strength_database.xlsx','CS (MPa)'),'SL':('slump_database.xlsx','SL (mm)')}.items():
    X,y,g=load(f,t); cols=list(X.columns); Xv=X.values; p=len(cols)
    rng=np.random.RandomState(1)
    outer=list(GroupKFold(5,shuffle=True,random_state=0).split(X,y,g))
    S=[];XS=[];base=[]
    for k,(tr,te) in enumerate(outer):
        m=ExtraTreesRegressor(n_estimators=100,random_state=0,n_jobs=1).fit(Xv[tr],y[tr])
        idx=rng.choice(te,min(NEXPL,len(te)),replace=False)
        phi=np.zeros((len(idx),p)); bg=Xv[tr]
        for i,ix in enumerate(idx):
            x=Xv[ix]; rows=[]; perms=[]
            for _ in range(NPERM):
                perm=rng.permutation(p); z=bg[rng.randint(len(bg))].copy(); perms.append(perm)
                rows.append(z.copy())
                for j in perm: z[j]=x[j]; rows.append(z.copy())
            pr=m.predict(np.array(rows)).reshape(NPERM,p+1)
            for q,perm in enumerate(perms):
                d=np.diff(pr[q]); phi[i,perm]+=d
            phi[i]/=NPERM
        S.append(phi); XS.append(Xv[idx]); base.append(m.predict(bg[:200]).mean())
        print(tname,k,flush=True)
    out[tname]=dict(cols=cols,phi=np.vstack(S).tolist(),X=np.vstack(XS).tolist())
json.dump(out,open('/tmp/ms/shap_mc.json','w'))
