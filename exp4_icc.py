import numpy as np, json
rng=np.random.RandomState(0); B=2000; out={}
def icc1(v,gi,G):
    n=np.bincount(gi,minlength=G).astype(float); s=np.bincount(gi,v,G); m=s/n; N=n.sum(); gm=v.mean()
    msb=(n*(m-gm)**2).sum()/(G-1); ssw=((v-m[gi])**2).sum(); msw=ssw/(N-G); n0=(N-(n**2).sum()/N)/(G-1)
    return (msb-msw)/(msb+(n0-1)*msw), N/G
for tn in ('CS','SL'):
    z=np.load(f'exp1_{tn}.npz'); y=z['y']; gi=z['gi']; G=gi.max()+1; r={}
    ic,mbar=icc1(y,gi,G); de=1+(mbar-1)*ic
    bs=[]
    for _ in range(B):
        s=rng.randint(0,G,G); idx=np.concatenate([np.where(gi==k)[0] for k in s]); g2=np.concatenate([[j]*(gi==k).sum() for j,k in enumerate(s)]).astype(int)
        bs.append(icc1(y[idx],g2,G)[0])
    r['icc_y']=ic; r['icc_y_ci']=np.percentile(bs,[2.5,97.5]).tolist(); r['mean_rows_per_source']=mbar; r['design_effect']=de; r['n_eff']=len(y)/de
    for sch in ('random','grouped'):
        P=z[f'base_{sch}']; shares=[]; iccs=[]
        for p in P:
            e=y-p; n=np.bincount(gi,minlength=G); b=np.bincount(gi,e,G)/n
            shares.append((n*b**2).sum()/(e**2).sum()); iccs.append(icc1(e,gi,G)[0])
        # bootstrap bias share for rep-averaged shares
        sh=[]
        for _ in range(B):
            w=np.bincount(rng.randint(0,G,G),minlength=G).astype(float); v=[]
            for p in P[:5]:
                e=y-p; n=np.bincount(gi,minlength=G); bb=np.bincount(gi,e,G)/n
                v.append((w*n*bb**2).sum()/(w*np.bincount(gi,e**2,G)).sum())
            sh.append(np.mean(v))
        r[sch]=dict(bias_share=float(np.mean(shares)),bias_share_ci=np.percentile(sh,[2.5,97.5]).tolist(),icc_resid=float(np.mean(iccs)))
    out[tn]=r
json.dump(out,open('exp4.json','w'),indent=1); print(json.dumps(out,indent=1))
