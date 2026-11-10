from common import *
ALPHA=0.1; R=40
def qh(s,a=ALPHA):
    s=np.sort(s); n=len(s); k=int(np.ceil((n+1)*(1-a)))-1; return s[min(k,n-1)]
def qw(s,w,a=ALPHA):
    o=np.argsort(s); s=s[o]; w=w[o]; wt=w.mean(); c=np.cumsum(w)/(w.sum()+wt)
    k=np.searchsorted(c,1-a); return s[min(k,len(s)-1)]
def cov(y,p,q,gte):
    h=np.abs(y-p)<=q; ps=pd.Series(h).groupby(gte).mean().values
    return h.mean(), 2*q, ps
out=[]; persrc={}
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs=np.unique(g); n=len(y)
    for rep in range(R):
        rng=np.random.RandomState(rep)
        # A
        pi=rng.permutation(n); a,b=int(.6*n),int(.8*n); tr,ca,te=pi[:a],pi[a:b],pi[b:]
        m=ET(150,rep).fit(Xv[tr],y[tr]); q=qh(np.abs(y[ca]-m.predict(Xv[ca])))
        c,w,_=cov(y[te],m.predict(Xv[te]),q,g[te]); out.append((tn,rep,'A',c,w,np.nan,np.nan))
        # sources
        ps=rng.permutation(srcs); s1,s2=int(.6*len(ps)),int(.8*len(ps))
        trs,cas,tes=ps[:s1],ps[s1:s2],ps[s2:]
        mtr=np.isin(g,trs); mca=np.isin(g,cas); mte=np.isin(g,tes); mpool=mtr|mca
        # B
        idx=np.where(mpool)[0]; idx=rng.permutation(idx); k=int(.75*len(idx)); btr,bca=idx[:k],idx[k:]
        m=ET(150,rep).fit(Xv[btr],y[btr]); q=qh(np.abs(y[bca]-m.predict(Xv[bca])))
        c,w,ps_=cov(y[mte],m.predict(Xv[mte]),q,g[mte]); out.append((tn,rep,'B',c,w,(ps_<.8).mean(),(ps_<.5).mean()))
        # C and D
        m=ET(150,rep).fit(Xv[mtr],y[mtr]); res=np.abs(y[mca]-m.predict(Xv[mca])); pte=m.predict(Xv[mte])
        c,w,ps_=cov(y[mte],pte,qh(res),g[mte]); out.append((tn,rep,'C',c,w,(ps_<.8).mean(),(ps_<.5).mean()))
        cnt=pd.Series(g[mca]).map(pd.Series(g[mca]).value_counts()).values; wts=1.0/cnt
        c,w,ps_=cov(y[mte],pte,qw(res,wts),g[mte]); out.append((tn,rep,'D',c,w,(ps_<.8).mean(),(ps_<.5).mean()))
        # E: grouped cross-conformal on 80% sources
        ip=np.where(mpool)[0]; oof=np.zeros(len(ip))
        for a_,b_ in GroupKFold(5,shuffle=True,random_state=rep).split(ip,groups=g[ip]):
            oof[b_]=ET(150,rep).fit(Xv[ip[a_]],y[ip[a_]]).predict(Xv[ip[b_]])
        res=np.abs(y[ip]-oof); mf=ET(150,rep).fit(Xv[ip],y[ip]); pte=mf.predict(Xv[mte])
        c,w,ps_=cov(y[mte],pte,qh(res),g[mte]); out.append((tn,rep,'E',c,w,(ps_<.8).mean(),(ps_<.5).mean()))
        cnt=pd.Series(g[ip]).map(pd.Series(g[ip]).value_counts()).values
        c,w,ps_=cov(y[mte],pte,qw(res,1.0/cnt),g[mte]); out.append((tn,rep,'F',c,w,(ps_<.8).mean(),(ps_<.5).mean()))
    print(tn,'done',flush=True)
df=pd.DataFrame(out,columns=['t','rep','scheme','cov','width','sub80','sub50']); df.to_csv('exp3.csv',index=False)
print(df.groupby(['t','scheme']).agg(['mean','std']).round(3))
