import pandas as pd, numpy as np, json
from sklearn.metrics import r2_score
res={}
def stats(s):
    # per-rep pooled metrics averaged over reps
    out=[]
    for rep,r in s.groupby('rep'):
        e0=r.y-r.p; e1=r.y-r.pa; hb=(e0.abs()<=r.qb); ha=(e1.abs()<=r.qa)
        sh=lambda e:(r.assign(e=e).groupby('g').e.agg(['size','mean']).pipe(lambda d:(d['size']*d['mean']**2).sum())/(e**2).sum())
        gb=hb.groupby(r.g).mean(); ga=ha.groupby(r.g).mean()
        out.append(dict(r2b=r2_score(r.y,r.p),r2a=r2_score(r.y,r.pa),rmb=np.sqrt((e0**2).mean()),rma=np.sqrt((e1**2).mean()),
          shb=sh(e0),sha=sh(e1),covb=hb.mean(),cova=ha.mean(),wb=2*r.qb.mean(),wa=2*r.qa.mean(),s80b=(gb<.8).mean(),s80a=(ga<.8).mean()))
    return pd.DataFrame(out).mean().to_dict()
for tn in ['CS','SL']:
    d=pd.read_csv(f'exp11_{tn}.csv'); srcs=d.g.unique()
    for proto,s in d.groupby('proto'):
        pt=stats(s); rng=np.random.RandomState(0); bs=[]
        for b in range(300):
            ch=rng.choice(srcs,len(srcs)); 
            ss=pd.concat([s[s.g==c].assign(g=f'{c}_{i}') for i,c in enumerate(ch)]) if False else None
            # fast bootstrap: weights per source
            w=pd.Series(ch).value_counts(); r_all=s.assign(wt=s.g.map(w).fillna(0))
            r_all=r_all[r_all.wt>0]; r_all=r_all.loc[r_all.index.repeat(r_all.wt.astype(int))]
            r_all=r_all.assign(g=r_all.g+'_'+r_all.groupby(level=0).cumcount().astype(str))
            # only use 3 reps for bootstrap speed
            r_all=r_all[r_all.rep<3]; bs.append(stats(r_all))
        bs=pd.DataFrame(bs)
        res[f'{tn}|{proto}']={k:dict(mean=pt[k],lo=float(bs[k].quantile(.025)),hi=float(bs[k].quantile(.975))) for k in pt}
        # delta R2 CI
        dd=(bs.r2a-bs.r2b); res[f'{tn}|{proto}']['d_r2']=dict(mean=pt['r2a']-pt['r2b'],lo=float(dd.quantile(.025)),hi=float(dd.quantile(.975)))
        dw=1-bs.wa/bs.wb; res[f'{tn}|{proto}']['d_w']=dict(mean=1-pt['wa']/pt['wb'],lo=float(dw.quantile(.025)),hi=float(dw.quantile(.975)))
        p=res[f'{tn}|{proto}']
        print(tn,proto,'n_eval',len(s)//20,'R2 %.3f->%.3f (d %.3f [%.3f,%.3f])'%(pt['r2b'],pt['r2a'],p['d_r2']['mean'],p['d_r2']['lo'],p['d_r2']['hi']),
              'RMSE %.1f->%.1f'%(pt['rmb'],pt['rma']),'share %.2f->%.2f'%(pt['shb'],pt['sha']),'cov %.3f->%.3f'%(pt['covb'],pt['cova']),
              'width %.1f->%.1f (%.0f%% [%.0f,%.0f])'%(pt['wb'],pt['wa'],100*p['d_w']['mean'],100*p['d_w']['lo'],100*p['d_w']['hi']),'s80 %.2f->%.2f'%(pt['s80b'],pt['s80a']),flush=True)
json.dump(res,open('ana11.json','w'),indent=1)
