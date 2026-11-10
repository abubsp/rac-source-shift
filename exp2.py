from common import *
from scipy.stats import spearmanr
X,y,g=load(*TG['SL']); Xv=X.values; R=20
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
rows=[]
for r in range(R):
    Pr=np.zeros(len(y)); Pl=np.zeros(len(y))
    for tr,te in GroupKFold(5,shuffle=True,random_state=r).split(Xv,y,g):
        Pr[te]=ET(100,r).fit(Xv[tr],y[tr]).predict(Xv[te])
        Pl[te]=np.expm1(ET(100,r).fit(Xv[tr],np.log1p(y[tr])).predict(Xv[te]))
    rows.append(dict(r2_raw=r2(y,Pr),r2_logmodel=r2(y,Pl),r2_logscale_raw=r2(np.log1p(y),np.log1p(np.clip(Pr,0,None))),
        r2_logscale_logmodel=r2(np.log1p(y),np.log1p(np.clip(Pl,0,None))),rho_raw=spearmanr(y,Pr)[0],rho_log=spearmanr(y,Pl)[0],
        rmse_raw=((y-Pr)**2).mean()**.5,rmse_log=((y-Pl)**2).mean()**.5))
df=pd.DataFrame(rows); print(df.mean().round(3)); print(df.std().round(3)); df.to_csv('exp2.csv',index=False)
