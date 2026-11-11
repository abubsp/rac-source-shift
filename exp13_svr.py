"""SVR with C and epsilon tuned INSIDE the folds (nested source-grouped), to remove the full-data tuning leak."""
from common import *
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import itertools
out={}
import sys
for tn,(f,t) in [(sys.argv[1],TG[sys.argv[1]])]:
    X,y,g=load(f,t); sd=y.std(); grid=list(itertools.product([1,10,100,1000],[0.002*sd,0.02*sd,0.1*sd]))
    r=[]
    for rep in range(2):
        p=np.zeros(len(y))
        for tr,te in GroupKFold(5,shuffle=True,random_state=rep).split(X,y,g):
            Xt,yt,gt=X.iloc[tr],y[tr],g[tr]; sc=[]
            for C,e in grid:
                pr=np.zeros(len(tr))
                for a,b in GroupKFold(3,shuffle=True,random_state=rep).split(Xt,yt,gt):
                    pr[b]=make_pipeline(StandardScaler(),SVR(C=C,epsilon=e)).fit(Xt.iloc[a],yt[a]).predict(Xt.iloc[b])
                sc.append(mean_squared_error(yt,pr))
            C,e=grid[int(np.argmin(sc))]; p[te]=make_pipeline(StandardScaler(),SVR(C=C,epsilon=e)).fit(Xt,yt).predict(X.iloc[te])
        r.append((r2_score(y,p),mean_squared_error(y,p)**.5))
        print(tn,rep,r[-1],flush=True)
    out[tn]=dict(r2=float(np.mean([a[0] for a in r])),rmse=float(np.mean([a[1] for a in r])))
json.dump(out,open(f'exp13_{sys.argv[1]}.json','w')); print(out)
