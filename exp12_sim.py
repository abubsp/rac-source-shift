"""Simulation: optimism of random CV (random R2 - source-grouped R2) as a function of the target ICC and of how tightly
the inputs of one source cluster (mean ICC of the inputs). Hierarchical synthetic data; 110 sources x 15 rows."""
from common import *
from sklearn.metrics import r2_score
import sys
def icc(y,g):
    df=pd.DataFrame(dict(y=y,g=g)); s=df.groupby('g').y.agg(['size','mean']); N=len(y); k=len(s)
    mw=((y-df.g.map(s['mean']).values)**2).sum()/(N-k); mb=(s['size']*(s['mean']-y.mean())**2).sum()/(k-1)
    m0=(N-(s['size']**2).sum()/N)/(k-1); return max((mb-mw)/(mb+(m0-1)*mw),0.0)
def gen(S,m,tau,w,rng,p=5):
    g=np.repeat(np.arange(S),m); mu=rng.normal(0,1,(S,p)); X=mu[g]+rng.normal(0,w,(S*m,p))
    f=2*X[:,0]+2*np.sin(2*X[:,1])+X[:,2]*X[:,3]+X[:,4]**2*0.5
    return X,f+rng.normal(0,tau,S)[g]+rng.normal(0,1,S*m),g
w=float(sys.argv[1]); out=[]; m=15; S=110
for tau in [0,2,4,8]:
    for rep in range(6):
        rng=np.random.RandomState(rep+int(1000*w)+int(10*tau)); X,y,g=gen(S,m,tau,w,rng)
        pr=np.zeros(len(y)); pg=np.zeros(len(y))
        for a,b in KFold(5,shuffle=True,random_state=rep).split(X): pr[b]=ET(100,rep).fit(X[a],y[a]).predict(X[b])
        for a,b in GroupKFold(5,shuffle=True,random_state=rep).split(X,y,g): pg[b]=ET(100,rep).fit(X[a],y[a]).predict(X[b])
        ic=icc(y,g); fi=np.mean([icc(X[:,j],g) for j in range(X.shape[1])])
        out.append((w,tau,rep,ic,fi,1+(m-1)*ic,r2_score(y,pr),r2_score(y,pg)))
d=pd.DataFrame(out,columns=['w','tau','rep','icc','feat_icc','de','r2_random','r2_grouped']); d['gap']=d.r2_random-d.r2_grouped
d.to_csv(f'exp12_w{sys.argv[1]}.csv',index=False)
print(d.groupby(['w','tau'])[['icc','feat_icc','r2_random','r2_grouped','gap']].mean().round(3))
