from common import *
R=20; out={}
for tn,(f,t) in TG.items():
    X0,y,g=load(f,t); X1=eng(X0); srcs,gi=np.unique(g,return_inverse=True)
    res={}
    for fs,X in (('base',X0),('eng',X1)):
        Xv=X.values
        for sch in ('random','grouped'):
            P=np.zeros((R,len(y)))
            for r in range(R):
                spl=(KFold(5,shuffle=True,random_state=r).split(Xv) if sch=='random' else GroupKFold(5,shuffle=True,random_state=r).split(Xv,y,g))
                for tr,te in spl: P[r,te]=ET(100,r).fit(Xv[tr],y[tr]).predict(Xv[te])
            res[f'{fs}|{sch}']=P
            print(tn,fs,sch,np.mean([1-((y-p)**2).sum()/((y-y.mean())**2).sum() for p in P]),flush=True)
    np.savez(f'exp1_{tn}.npz',y=y,gi=gi,**{k.replace('|','_'):v for k,v in res.items()})
