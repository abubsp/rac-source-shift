from common import *
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
ALPHA=.1; R=40
def qh(s,a=ALPHA):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*(1-a)))-1; return s[min(k,n-1)]
def tstd(m,X):
    A=np.stack([t.predict(X.astype(np.float32)) for t in m.estimators_]); return A.mean(0),A.std(0)
def knn(Xtr,Xq,k=10):
    sc=StandardScaler().fit(Xtr); nn=NearestNeighbors(n_neighbors=k).fit(sc.transform(Xtr)); d,_=nn.kneighbors(sc.transform(Xq)); return d.mean(1)
out=[]
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs=np.unique(g); n=len(y)
    for rep in range(R):
        rng=np.random.RandomState(rep); rng.permutation(n); ps=rng.permutation(srcs)
        s2=int(.8*len(ps)); tes=ps[s2:]; mte=np.isin(g,tes); ip=np.where(~mte)[0]; te=np.where(mte)[0]
        folds=list(GroupKFold(5,shuffle=True,random_state=rep).split(ip,groups=g[ip]))
        oof=np.zeros(len(ip)); so=np.zeros(len(ip)); ko=np.zeros(len(ip)); fid=np.zeros(len(ip),int); Mte=np.zeros((5,len(te)))
        for k,(a_,b_) in enumerate(folds):
            m=ET(150,rep).fit(Xv[ip[a_]],y[ip[a_]]); oof[b_],so[b_]=tstd(m,Xv[ip[b_]]); ko[b_]=knn(Xv[ip[a_]],Xv[ip[b_]]); fid[b_]=k; Mte[k]=m.predict(Xv[te])
        mf=ET(150,rep).fit(Xv[ip],y[ip]); pte,ste=tstd(mf,Xv[te]); kte=knn(Xv[ip],Xv[te])
        res=np.abs(y[ip]-oof); yt=y[te]; et=np.abs(yt-pte)
        # error-model scale (cross-fitted by source folds)
        lo=np.zeros(len(ip))
        for a_,b_ in folds: lo[b_]=ET(100,rep).fit(Xv[ip[a_]],res[a_]).predict(Xv[ip[b_]])
        lte=ET(100,rep).fit(Xv[ip],res).predict(Xv[te]); lo=np.clip(lo,1e-6,None); lte=np.clip(lte,1e-6,None)
        def rec(name,lo_,hi_):
            h=(yt>=lo_)&(yt<=hi_); ps_=pd.Series(h).groupby(g[te]).mean().values
            isc=np.mean((hi_-lo_)+(2/ALPHA)*np.maximum(lo_-yt,0)+(2/ALPHA)*np.maximum(yt-hi_,0))
            out.append((tn,rep,name,h.mean(),(hi_-lo_).mean(),(ps_<.8).mean(),isc))
        for name,sc_o,sc_t in (('E_plain',np.ones(len(ip)),np.ones(len(te))),('G_treestd',so,ste),('I_both',so/np.median(so)+ko/np.median(ko),ste/np.median(so)+kte/np.median(ko)),('L_errormodel',lo,lte)):
            eps=0.1*np.median(sc_o); q=qh(res/(sc_o+eps)); hw=q*(sc_t+eps); rec(name,pte-hw,pte+hw)
        # CV+ with source-grouped folds
        nn_=len(ip); kk=int(np.floor(ALPHA*(nn_+1)))-1; ku=int(np.ceil((1-ALPHA)*(nn_+1)))-1
        Lw=Mte[fid]-res[:,None]; Up=Mte[fid]+res[:,None]
        lo_=np.sort(Lw,axis=0)[max(kk,0)]; hi_=np.sort(Up,axis=0)[min(ku,nn_-1)]; rec('J_groupCVplus',lo_,hi_)
    print(tn,'done',flush=True)
df=pd.DataFrame(out,columns=['t','rep','scheme','cov','width','sub80','iscore']); df.to_csv('exp8.csv',index=False)
print(df.groupby(['t','scheme'])[['cov','width','sub80','iscore']].mean().round(3).to_string())
