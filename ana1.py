import numpy as np, json
B=2000; rng=np.random.RandomState(0); res={}
def stats(y,gi,P,G):
    # per-source sufficient stats per rep: n, sse ; global y stats per source
    n=np.bincount(gi,minlength=G).astype(float); sy=np.bincount(gi,y,G); syy=np.bincount(gi,y*y,G)
    sse=np.stack([np.bincount(gi,(y-p)**2,G) for p in P]); return n,sy,syy,sse
for tn in ('CS','SL'):
    z=np.load(f'exp1_{tn}.npz'); y=z['y']; gi=z['gi']; G=gi.max()+1
    S={k:stats(y,gi,z[k],G) for k in ['base_random','base_grouped','eng_random','eng_grouped']}
    n,sy,syy,_=S['base_random']
    def R2(k,w):  # mean over reps of pooled R2 on bootstrap weights w
        n,sy,syy,sse=S[k]; N=(w*n).sum(); sst=(w*syy).sum()-((w*sy).sum())**2/N
        return (1-(sse*w).sum(1)/sst)  # per-rep
    W=[np.bincount(rng.randint(0,G,G),minlength=G).astype(float) for _ in range(B)]
    W0=np.ones(G)
    out={}
    for k in S:
        r=R2(k,W0); bs=np.array([R2(k,w).mean() for w in W])
        out[k]=dict(mean=r.mean(),sd_reps=r.std(ddof=1),ci=np.percentile(bs,[2.5,97.5]).tolist(),
                    rmse=float(np.mean([ (S[k][3][i].sum()/S[k][0].sum())**.5 for i in range(S[k][3].shape[0])])))
    def diff(k1,k2):
        d0=(R2(k1,W0)-R2(k2,W0)).mean(); bs=np.array([(R2(k1,w)-R2(k2,w)).mean() for w in W])
        return dict(mean=d0,ci=np.percentile(bs,[2.5,97.5]).tolist(),p_le0=float((bs<=0).mean()))
    out['gap_base']=diff('base_random','base_grouped'); out['eng_minus_base_grouped']=diff('eng_grouped','base_grouped'); out['eng_minus_base_random']=diff('eng_random','base_random')
    res[tn]=out
json.dump(res,open('ana1.json','w'),indent=1)
for tn in res:
    print(tn)
    for k,v in res[tn].items(): print(' ',k,{a:(np.round(b,3).tolist() if not isinstance(b,float) else round(b,3)) for a,b in v.items()})
