"""Source-offset adaptation: use a few measured mixtures of a NEW source (e.g. its natural-aggregate controls)
to estimate that source's offset, shrink it (empirical Bayes), and re-centre predictions + conformal intervals.
Evaluation is always on mixtures of the new source that were NOT used as anchors."""
from common import *
import sys
ALPHA=.1; R=int(sys.argv[2]) if len(sys.argv)>2 else 20
def qh(s,a=ALPHA):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*(1-a)))-1; return s[min(k,n-1)]
def varcomp(e,g):
    df=pd.DataFrame(dict(e=e,g=g)); s=df.groupby('g').e.agg(['size','mean']); N=len(e); k=len(s)
    mw=((e-df.g.map(s['mean']).values)**2).sum()/(N-k); gm=e.mean()
    mb=(s['size']*(s['mean']-gm)**2).sum()/(k-1); m0=(N-(s['size']**2).sum()/N)/(k-1)
    return max((mb-mw)/m0,0.0),mw
def anchors(proto,idx,rar,rng):
    """indices (positions within idx) used as anchors for one source"""
    if proto.startswith('ctrl'):
        K=int(proto[4:]); c=np.where(rar[idx]==0)[0]
        if len(c)==0: return None
        return rng.permutation(c)[:K]
    k=int(proto[4:]); 
    if len(idx)<=k+2: return None
    return rng.permutation(len(idx))[:k]
PROTOS=['ctrl1','ctrl2','ctrl3','rand1','rand3']
tn=sys.argv[1]; f,t=TG[tn]; X,y,g=load(f,t); Xv=X.values; rar=X.rar.values; srcs=np.unique(g)
hasctl=pd.Series(rar==0).groupby(g).any(); ctlsrc=set(hasctl[hasctl].index)
rows=[]
for rep in range(R):
    for fold,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=rep).split(Xv,y,g)):
        rng=np.random.RandomState(1000*rep+fold)
        m=ET(100,rep).fit(Xv[tr],y[tr]); pte=m.predict(Xv[te])
        oof=np.zeros(len(tr))
        for a,b in GroupKFold(5,shuffle=True,random_state=rep).split(tr,groups=g[tr]):
            oof[b]=ET(100,rep).fit(Xv[tr[a]],y[tr[a]]).predict(Xv[tr[b]])
        etr=y[tr]-oof; tau2,sig2=varcomp(etr,g[tr])
        gtr=g[tr]; gte=g[te]
        for proto in PROTOS:
            # calibration (training sources having a control, non-anchor rows)
            rb=[];ra=[]
            for s in np.unique(gtr):
                if s not in ctlsrc: continue
                ii=np.where(gtr==s)[0]; an=anchors(proto,ii,rar[tr],rng)
                if an is None: continue
                ev=np.setdiff1d(np.arange(len(ii)),an); ev=ev[rar[tr][ii[ev]]>0]
                if len(ev)==0: continue
                na=len(an); w=tau2/(tau2+sig2/na); d=w*etr[ii[an]].mean()
                rb.append(np.abs(etr[ii[ev]])); ra.append(np.abs(etr[ii[ev]]-d))
            qb=qh(np.concatenate(rb)); qa=qh(np.concatenate(ra))
            for s in np.unique(gte):
                if s not in ctlsrc: continue
                ii=np.where(gte==s)[0]; an=anchors(proto,ii,rar[te],rng)
                if an is None: continue
                ev=np.setdiff1d(np.arange(len(ii)),an); ev=ev[rar[te][ii[ev]]>0]
                if len(ev)==0: continue
                ea=y[te][ii[an]]-pte[ii[an]]; na=len(an); w=tau2/(tau2+sig2/na); d=w*ea.mean(); draw=ea.mean()
                for j in ev:
                    yy=y[te][ii[j]]; p=pte[ii[j]]
                    rows.append((tn,rep,proto,s,yy,p,p+d,p+draw,qb,qa))
    print(tn,rep,flush=True)
df=pd.DataFrame(rows,columns=['t','rep','proto','g','y','p','pa','praw','qb','qa']); df.to_csv(f'exp11_{tn}.csv',index=False)
