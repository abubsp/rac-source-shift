from common import *
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
ALPHA=.1; R=40
def qh(s,a=ALPHA):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*(1-a)))-1; return s[min(k,n-1)]
def tstd(m,X):
    A=np.stack([t.predict(X.astype(np.float32)) for t in m.estimators_]); return A.mean(0),A.std(0)
def knn(Xtr,Xq,k=10):
    sc=StandardScaler().fit(Xtr); nn=NearestNeighbors(n_neighbors=k).fit(sc.transform(Xtr)); d,_=nn.kneighbors(sc.transform(Xq)); return d.mean(1)
out=[]; cnd=[]; sp=[]
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs=np.unique(g); n=len(y)
    for rep in range(R):
        rng=np.random.RandomState(rep); rng.permutation(n); ps=rng.permutation(srcs)
        s1,s2=int(.6*len(ps)),int(.8*len(ps)); tes=ps[s2:]; mte=np.isin(g,tes); ip=np.where(~mte)[0]; te=np.where(mte)[0]
        oof=np.zeros(len(ip)); so=np.zeros(len(ip)); ko=np.zeros(len(ip))
        for a_,b_ in GroupKFold(5,shuffle=True,random_state=rep).split(ip,groups=g[ip]):
            m=ET(150,rep).fit(Xv[ip[a_]],y[ip[a_]]); oof[b_],so[b_]=tstd(m,Xv[ip[b_]]); ko[b_]=knn(Xv[ip[a_]],Xv[ip[b_]])
        mf=ET(150,rep).fit(Xv[ip],y[ip]); pte,ste=tstd(mf,Xv[te]); kte=knn(Xv[ip],Xv[te])
        res=np.abs(y[ip]-oof); yt=y[te]; et=np.abs(yt-pte)
        for name,sc_o,sc_t in (('E_plain',np.ones(len(ip)),np.ones(len(te))),('G_treestd',so,ste),('H_knn',ko,kte),('I_both',so/np.median(so)+ko/np.median(ko),ste/np.median(so)+kte/np.median(ko))):
            eps=0.1*np.median(sc_o); q=qh(res/(sc_o+eps)); hw=q*(sc_t+eps)
            h=et<=hw; psrc=pd.Series(h).groupby(g[te]).mean().values
            iscore=np.mean(2*hw+(2/ALPHA)*np.maximum(et-hw,0))
            # tertiles of sigma by calibration cutoffs
            c1,c2=np.percentile(sc_o,[33.3,66.7]); tt=np.digitize(sc_t,[c1,c2])
            ctr=[h[tt==j].mean() if (tt==j).sum()>10 else np.nan for j in range(3)]
            wtr=[(2*hw)[tt==j].mean() if (tt==j).sum()>10 else np.nan for j in range(3)]
            out.append((tn,rep,name,h.mean(),(2*hw).mean(),(psrc<.8).mean(),iscore,*ctr,*wtr))
            if name=='G_treestd' or name=='H_knn': sp.append((tn,rep,name,spearmanr(sc_t,et)[0]))
    print(tn,'done',flush=True)
cols=['t','rep','scheme','cov','width','sub80','iscore','cov_t1','cov_t2','cov_t3','w_t1','w_t2','w_t3']
df=pd.DataFrame(out,columns=cols); df.to_csv('exp6.csv',index=False); pd.DataFrame(sp,columns=['t','rep','name','rho']).to_csv('exp6_rho.csv',index=False)
print(df.groupby(['t','scheme'])[cols[3:]].mean().round(3).to_string())
print(pd.read_csv('exp6_rho.csv').groupby(['t','name']).rho.mean().round(3))
