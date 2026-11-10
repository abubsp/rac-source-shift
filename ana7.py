import numpy as np, json
B=2000; rng=np.random.RandomState(0); out={}
for tn in ('CS','SL'):
    z=np.load(f'exp7_{tn}.npz'); y=z['y']; gi=z['gi']; G=gi.max()+1; n=np.bincount(gi,minlength=G).astype(float); sy=np.bincount(gi,y,G); syy=np.bincount(gi,y*y,G)
    W=[np.bincount(rng.randint(0,G,G),minlength=G).astype(float) for _ in range(B)]; W0=np.ones(G)
    for L in ('RF','HGB','XGB','LGBM'):
        S={}
        for sch in ('random','grouped'):
            P=z[f'{L}_{sch}']; S[sch]=(np.stack([np.bincount(gi,(y-p)**2,G) for p in P]),np.stack([np.bincount(gi,y-p,G) for p in P]))
        def R2(sch,w): sse=S[sch][0]; N=(w*n).sum(); sst=(w*syy).sum()-((w*sy).sum())**2/N; return 1-(sse*w).sum(1)/sst
        def share(sch,w): sse,se=S[sch]; return ((w*se**2/np.maximum(n,1)).sum(1))/((w*sse).sum(1))
        r=dict()
        for sch in ('random','grouped'):
            bs=np.array([R2(sch,w).mean() for w in W]); r[sch+'_r2']=[R2(sch,W0).mean(),*np.percentile(bs,[2.5,97.5])]
            bs2=np.array([share(sch,w).mean() for w in W]); r[sch+'_share']=[share(sch,W0).mean(),*np.percentile(bs2,[2.5,97.5])]
        bs=np.array([(R2('random',w)-R2('grouped',w)).mean() for w in W]); r['gap']=[(R2('random',W0)-R2('grouped',W0)).mean(),*np.percentile(bs,[2.5,97.5])]
        out[f'{tn}|{L}']=r
        print(tn,L,{k:[round(x,3) for x in v] for k,v in r.items()})
json.dump(out,open('ana7.json','w'),indent=1)
