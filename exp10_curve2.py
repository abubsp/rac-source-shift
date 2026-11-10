from common import *
R=20; KS={'CS':[10,20,35,50,70,88],'SL':[8,15,25,35,45,52]}; B=2000; brng=np.random.RandomState(0); out={}
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs,gi=np.unique(g,return_inverse=True); G=len(srcs); K=KS[tn]
    SSE=np.zeros((R,len(K),G))
    for rep in range(R):
        rng=np.random.RandomState(2000+rep); P=np.zeros((len(K),len(y)))
        for tr,te in GroupKFold(5,shuffle=True,random_state=rep).split(Xv,y,g):
            pool=np.unique(gi[tr]); pool=rng.permutation(pool)
            for j,k in enumerate(K):
                sel=np.isin(gi,pool[:min(k,len(pool))]); P[j,te]=ET(100,rep).fit(Xv[sel],y[sel]).predict(Xv[te])
        for j in range(len(K)): SSE[rep,j]=np.bincount(gi,(y-P[j])**2,G)
        print(tn,rep,flush=True) if rep%5==0 else None
    n=np.bincount(gi,minlength=G).astype(float); sy=np.bincount(gi,y,G); syy=np.bincount(gi,y*y,G)
    def r2(w): N=(w*n).sum(); sst=(w*syy).sum()-((w*sy).sum())**2/N; return 1-(SSE*w).sum(2)/sst   # R x K
    r0=r2(np.ones(G)); bs=np.array([r2(np.bincount(brng.randint(0,G,G),minlength=G).astype(float)).mean(0) for _ in range(B)])
    out[tn]=dict(k=K,mean=r0.mean(0).tolist(),sd_reps=r0.std(0,ddof=1).tolist(),ci_lo=np.percentile(bs,2.5,axis=0).tolist(),ci_hi=np.percentile(bs,97.5,axis=0).tolist())
    print(tn,out[tn],flush=True)
import json; json.dump(out,open('exp10.json','w'),indent=1)
