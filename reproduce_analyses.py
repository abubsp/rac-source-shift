"""Re-implementation of the main analyses of the manuscript (Sections 3.5-3.8 / 4.1-4.7), written from the
methods text. Usage: python reproduce_analyses.py [table3 loso perm joint conformal paired pd]
Set DATA_DIR to the folder holding the two .xlsx files. Results go to results_<name>.json"""
import sys, json, numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
DATA_DIR=os.environ.get('RAC_DATA_DIR','.')
IN=['ρ-RCA (kg/m3)','CV-RCA (%)','W-RCA (%)','Cement (kg/m3)','FA (kg/m3)','GGBS (kg/m3)','Water (kg/m3)','Sand (kg/m3)','CA (kg/m3)','RAR (%)','SP (kg/m3)']
SH=['rho','cv','wabs','cem','fa','ggbs','water','sand','ca','rar','sp']
def load(kind,unflagged=True):
    f,t={'CS':('compressive_strength_database.xlsx','CS (MPa)'),'SL':('slump_database.xlsx','SL (mm)')}[kind]
    d=pd.read_excel(f'{DATA_DIR}/{f}'); d['src']=d.source_reference.ffill()
    if unflagged: d=d[d.outlier_flag==0]
    d=d.reset_index(drop=True); X=d[IN].copy(); X.columns=SH
    return X,d[t].values,d.src.values
def eng(X):
    E=X.copy(); b=X.cem+X.fa+X.ggbs
    E['wb']=X.water/b; E['binder']=b; E['scm']=(X.fa+X.ggbs)/b; E['sp_b']=X.sp/b
    E['ew']=X.water+X.ca*X.rar/100*X.wabs/100; E['ewb']=E.ew/b; E['sandr']=X.sand/(X.sand+X.ca); E['paste']=b+X.water
    return E
def models(kind,seed=0):
    sv=dict(C=10,epsilon=0.1) if kind=='CS' else dict(C=100,epsilon=1.0)
    return {'Ridge':make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-3,3,13))),
     'SVR':make_pipeline(StandardScaler(),SVR(kernel='rbf',gamma='scale',**sv)),
     'MLP':TransformedTargetRegressor(make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,64),alpha=0.01,early_stopping=True,max_iter=800,random_state=seed)),transformer=StandardScaler()),
     'RF':RandomForestRegressor(300,min_samples_leaf=2,random_state=seed,n_jobs=1),
     'ET':ExtraTreesRegressor(250,random_state=seed,n_jobs=1),
     'GB':GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,max_depth=4,subsample=0.8,random_state=seed),
     'HGB':HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=10,random_state=seed)}
def oof(m,X,y,splits):
    p=np.zeros(len(y))
    for tr,te in splits: p[te]=m.fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
    return p
def rm(y,p): return float(mean_squared_error(y,p)**.5)
def table3():
    out={}
    for kind in ['CS','SL']:
        X,y,g=load(kind); XE=eng(X)
        for name in ['Ridge','SVR','MLP','RF','ET','GB','HGB','ET+eng','HGB+eng']:
            base=name.split('+')[0]; Xu=XE if '+eng' in name else X; r=[]
            for rep in range(2):
                m=models(kind,rep)[base]
                pr=oof(m,Xu,y,list(KFold(5,shuffle=True,random_state=rep).split(Xu)))
                pg=oof(models(kind,rep)[base],Xu,y,list(GroupKFold(5,shuffle=True,random_state=rep).split(Xu,y,g)))
                r.append((r2_score(y,pr),r2_score(y,pg),rm(y,pg)))
            out[f'{kind}|{name}']=dict(r2_random=float(np.mean([a[0] for a in r])),r2_grouped=float(np.mean([a[1] for a in r])),rmse_grouped=float(np.mean([a[2] for a in r])))
            print(kind,name,out[f'{kind}|{name}'],flush=True)
    return out
def loso():
    out={}
    for kind in ['CS','SL']:
        X,y,g=load(kind); p=np.zeros(len(y))
        for tr,te in LeaveOneGroupOut().split(X,y,g): p[te]=ExtraTreesRegressor(200,random_state=0,n_jobs=1).fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
        df=pd.DataFrame(dict(g=g,e=p-y)); s=df.groupby('g').e
        rmse=s.apply(lambda e:np.sqrt((e**2).mean())); bias=s.mean().abs()
        out[kind]=dict(n_sources=int(df.g.nunique()),r2=r2_score(y,p),rmse=rm(y,p),med_source_rmse=float(rmse.median()),p90_source_rmse=float(rmse.quantile(.9)),med_abs_bias=float(bias.median()))
        print(kind,out[kind],flush=True)
    return out
def perm():
    out={}
    for kind in ['CS','SL']:
        X,y,g=load(kind); rng=np.random.RandomState(0); inc=np.zeros(X.shape[1]); nf=0
        for tr,te in GroupKFold(5,shuffle=True,random_state=0).split(X,y,g):
            m=ExtraTreesRegressor(200,random_state=0,n_jobs=1).fit(X.iloc[tr],y[tr]); base=rm(y[te],m.predict(X.iloc[te]))
            for j,c in enumerate(X.columns):
                d=[]
                for _ in range(8):
                    Z=X.iloc[te].copy(); Z[c]=rng.permutation(Z[c].values); d.append(rm(y[te],m.predict(Z))-base)
                inc[j]+=np.mean(d)
            nf+=1
        out[kind]=dict(zip(SH,(inc/nf).round(2).tolist())); print(kind,out[kind],flush=True)
    return out
def joint():
    """Table 6: R2 and RMSE are means over the five folds and four repetitions (per-fold values), as in the manuscript."""
    from scipy.stats import spearmanr
    def matched(unflagged):
        Xc,yc,gc=load('CS',unflagged); Xs,ys,gs=load('SL',unflagged)
        a=Xc.round(3).assign(cs=yc,src=gc).groupby(SH,as_index=False).agg(cs=('cs','mean'),src=('src','first'))
        b=Xs.round(3).assign(sl=ys).groupby(SH,as_index=False).agg(sl=('sl','mean'))
        a[SH]=a[SH].round(3); b[SH]=b[SH].round(3); return a.merge(b,on=SH)
    d=matched(True); X=d[SH]; Y=d[['cs','sl']].values; g=d.src.values
    res={}
    for tag in ['ET','MLP']:
        mk=(lambda s:ExtraTreesRegressor(200,random_state=s,n_jobs=1)) if tag=='ET' else \
           (lambda s:TransformedTargetRegressor(make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,64),alpha=0.01,early_stopping=True,max_iter=800,random_state=s)),transformer=StandardScaler()))
        for mode in ['separate','joint']:
            M=[]
            for rep in range(4):
                for tr,te in GroupKFold(5,shuffle=True,random_state=rep).split(X,Y,g):
                    if mode=='separate': P=np.column_stack([mk(rep).fit(X.iloc[tr],Y[tr,k]).predict(X.iloc[te]) for k in range(2)])
                    else: P=mk(rep).fit(X.iloc[tr],Y[tr]).predict(X.iloc[te])
                    M.append([r2_score(Y[te,k],P[:,k]) for k in range(2)]+[rm(Y[te,k],P[:,k]) for k in range(2)])
            M=np.array(M); res[f'{tag}_{mode}']=dict(r2_cs=float(M[:,0].mean()),r2_sl=float(M[:,1].mean()),rmse_cs=float(M[:,2].mean()),rmse_sl=float(M[:,3].mean()),sd_r2_cs=float(M[:,0].std()),sd_r2_sl=float(M[:,1].std()))
            print(tag,mode,res[f'{tag}_{mode}'],flush=True)
    d_all=matched(False)
    res['n_mixtures']=len(d); res['n_sources']=int(d.src.nunique()); res['n_mixtures_all']=len(d_all); res['n_sources_all']=int(d_all.src.nunique())
    res['spearman_885']=float(spearmanr(d.cs,d.sl)[0]); res['spearman_937']=float(spearmanr(d_all.cs,d_all.sl)[0])
    print(res,flush=True); return res
def conformal(nrep=40):
    def q(s,a=0.1):
        n=len(s); k=int(np.ceil((n+1)*(1-a))); return np.sort(s)[min(k,n)-1]
    out={}
    for kind in ['CS','SL']:
        X,y,g=load(kind); us=np.unique(g); R={'A':[],'B':[],'C':[]}
        for rep in range(nrep):
            rng=np.random.RandomState(rep)
            # A: random rows 60/20/20
            i=rng.permutation(len(y)); a,b=int(.6*len(y)),int(.8*len(y)); tr,ca,te=i[:a],i[a:b],i[b:]
            m=ExtraTreesRegressor(150,random_state=rep,n_jobs=1).fit(X.iloc[tr],y[tr]); qq=q(np.abs(y[ca]-m.predict(X.iloc[ca])))
            e=np.abs(y[te]-m.predict(X.iloc[te])); R['A'].append((float((e<=qq).mean()),2*qq))
            # sources 60/20/20
            s=rng.permutation(us); n=len(s); S1,S2,S3=s[:int(.6*n)],s[int(.6*n):int(.8*n)],s[int(.8*n):]
            te=np.where(np.isin(g,S3))[0]
            pool=np.where(np.isin(g,np.r_[S1,S2]))[0]; pool=rng.permutation(pool); k=int(.75*len(pool)); tr,ca=pool[:k],pool[k:]
            m=ExtraTreesRegressor(150,random_state=rep,n_jobs=1).fit(X.iloc[tr],y[tr]); qq=q(np.abs(y[ca]-m.predict(X.iloc[ca])))
            e=np.abs(y[te]-m.predict(X.iloc[te])); R['B'].append((float((e<=qq).mean()),2*qq))
            tr=np.where(np.isin(g,S1))[0]; ca=np.where(np.isin(g,S2))[0]
            m=ExtraTreesRegressor(150,random_state=rep,n_jobs=1).fit(X.iloc[tr],y[tr]); qq=q(np.abs(y[ca]-m.predict(X.iloc[ca])))
            e=np.abs(y[te]-m.predict(X.iloc[te])); R['C'].append((float((e<=qq).mean()),2*qq))
        out[kind]={k:dict(cov=float(np.mean([a[0] for a in v])),cov_sd=float(np.std([a[0] for a in v])),width=float(np.mean([a[1] for a in v]))) for k,v in R.items()}
        print(kind,out[kind],flush=True)
    return out
def paired():
    X,y,g=load('CS'); d=X.assign(cs=y,src=g); d['wb']=d.water/(d.cem+d.fa+d.ggbs); rows=[]
    for s,grp in d.groupby('src'):
        ctl=grp[grp.rar==0]
        if ctl.empty: continue
        for _,r in grp[grp.rar>0].iterrows():
            c=ctl[(ctl.wb-r.wb).abs()<=0.02]
            if len(c): rows.append(dict(src=s,rar=r.rar,cv=r.cv,ratio=r.cs/c.cs.mean(),scm=(r.fa+r.ggbs)>0))
    P=pd.DataFrame(rows); out=dict(n_pairs=len(P),n_sources=int(P.src.nunique()))
    t1,t2=P.cv.quantile([1/3,2/3]); out['tertile_edges']=[float(P.cv.min()),float(t1),float(t2),float(P.cv.max())]
    P['tert']=np.where(P.cv<=t1,'low',np.where(P.cv<=t2,'mid','high')); F=P[P.rar>=99]; rng=np.random.RandomState(0)
    for t in ['low','mid','high']:
        f=F[F.tert==t]; srcs=f.src.unique(); bs=[]
        for _ in range(1000):
            ch=rng.choice(srcs,len(srcs)); bs.append(np.mean(np.concatenate([f.ratio[f.src==s].values for s in ch])))
        out[t]=dict(mean=float(f.ratio.mean()),lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)),pairs=len(f),sources=int(f.src.nunique()))
    print(out,flush=True); return out
def pd_curves():
    X,y,g=load('CS'); m=ExtraTreesRegressor(200,random_state=0,n_jobs=1).fit(X,y); med=X.median(); out={}
    for cvv in [10,15,20]:
        v=[]
        for rar in [0,100]:
            z=med.copy(); z['rar']=rar; z['cv']=cvv; z['wabs']=med.wabs+0.2*(cvv-med.cv)
            v.append(float(m.predict(pd.DataFrame([z]))[0]))
        out[f'CV{cvv}']=[round(a,1) for a in v]
    print(out,flush=True); return out
if __name__=='__main__':
    todo=sys.argv[1:] or ['table3','loso','perm','joint','conformal','paired','pd']
    F=dict(table3=table3,loso=loso,perm=perm,joint=joint,conformal=conformal,paired=paired,pd=pd_curves)
    for k in todo: json.dump(F[k](),open(f'results_{k}.json','w'),indent=1,default=float)
