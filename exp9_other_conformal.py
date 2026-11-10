from common import *
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
ALPHA=.1; R=20
def qh(s,a=ALPHA):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*(1-a)))-1; return s[min(k,n-1)]
def knn(Xtr,Xq,k=10):
    sc=StandardScaler().fit(Xtr); nn=NearestNeighbors(n_neighbors=k).fit(sc.transform(Xtr)); d,_=nn.kneighbors(sc.transform(Xq)); return d.mean(1)
def mk(n,r):
    if n=='RF': return RandomForestRegressor(150,min_samples_leaf=2,random_state=r,n_jobs=1)
    return HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10,random_state=r)
out=[]
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs=np.unique(g); n=len(y)
    for rep in range(R):
        rng=np.random.RandomState(rep); pi=rng.permutation(n); ps=rng.permutation(srcs)
        s1,s2=int(.6*len(ps)),int(.8*len(ps)); trs,cas,tes=ps[:s1],ps[s1:s2],ps[s2:]
        mtr=np.isin(g,trs); mca=np.isin(g,cas); mte=np.isin(g,tes); mpool=mtr|mca; yt=y[mte]; te=np.where(mte)[0]
        for ln in ('RF','HGB'):
            def rec(name,hw,pte):
                h=np.abs(yt-pte)<=hw; ps_=pd.Series(h).groupby(g[mte]).mean().values
                out.append((tn,ln,rep,name,h.mean(),(2*hw).mean(),(ps_<.8).mean(),np.mean(2*hw+(2/ALPHA)*np.maximum(np.abs(yt-pte)-hw,0))))
            # B: random rows within pool
            idx=rng.permutation(np.where(mpool)[0]); k=int(.75*len(idx)); btr,bca=idx[:k],idx[k:]
            m=mk(ln,rep).fit(Xv[btr],y[btr]); q=qh(np.abs(y[bca]-m.predict(Xv[bca]))); rec('B',np.full(len(te),q),m.predict(Xv[te]))
            # C
            m=mk(ln,rep).fit(Xv[mtr],y[mtr]); q=qh(np.abs(y[mca]-m.predict(Xv[mca]))); rec('C',np.full(len(te),q),m.predict(Xv[te]))
            # E and H
            ip=np.where(mpool)[0]; oof=np.zeros(len(ip)); ko=np.zeros(len(ip))
            for a_,b_ in GroupKFold(5,shuffle=True,random_state=rep).split(ip,groups=g[ip]):
                oof[b_]=mk(ln,rep).fit(Xv[ip[a_]],y[ip[a_]]).predict(Xv[ip[b_]]); ko[b_]=knn(Xv[ip[a_]],Xv[ip[b_]])
            res=np.abs(y[ip]-oof); mf=mk(ln,rep).fit(Xv[ip],y[ip]); pte=mf.predict(Xv[te]); kte=knn(Xv[ip],Xv[te])
            rec('E',np.full(len(te),qh(res)),pte)
            eps=0.1*np.median(ko); q=qh(res/(ko+eps)); rec('H',q*(kte+eps),pte)
    print(tn,'done',flush=True)
df=pd.DataFrame(out,columns=['t','learner','rep','scheme','cov','width','sub80','iscore']); df.to_csv('exp9.csv',index=False)
print(df.groupby(['t','learner','scheme'])[['cov','width','sub80','iscore']].mean().round(3).to_string())
