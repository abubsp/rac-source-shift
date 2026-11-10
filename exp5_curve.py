from common import *
R=40; res={}
KS={'CS':[10,20,35,50,70,89],'SL':[8,15,25,35,45,53]}
for tn,(f,t) in TG.items():
    X,y,g=load(f,t); Xv=X.values; srcs=np.unique(g); nt=int(round(.2*len(srcs)))
    rows=[]
    for rep in range(R):
        rng=np.random.RandomState(1000+rep); ps=rng.permutation(srcs); te=ps[:nt]; pool=ps[nt:]
        mte=np.isin(g,te)
        for k in KS[tn]:
            trs=pool[:k]; mtr=np.isin(g,trs)
            p=ET(100,rep).fit(Xv[mtr],y[mtr]).predict(Xv[mte]); yt=y[mte]
            rows.append((rep,k,1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum(),(((yt-p)**2).mean())**.5))
    df=pd.DataFrame(rows,columns=['rep','k','r2','rmse']); df.to_csv(f'exp5_{tn}.csv',index=False)
    print(tn); print(df.groupby('k').agg(r2=('r2','mean'),r2_sd=('r2','std'),r2_med=('r2','median'),rmse=('rmse','mean')).round(3),flush=True)
