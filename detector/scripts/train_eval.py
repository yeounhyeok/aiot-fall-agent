"""Train fall detector on synthetic (MuJoCo) IMU and evaluate on SisFall.
Features (orientation-invariant, 100 Hz, 3 s window): |a| [g], |w|/100 [deg/s], tilt vs window start [rad].
"""
import numpy as np, torch, torch.nn as nn

DEV=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS=100; W=300; STRIDE=25

def down(x,f):  # block-mean resample
    n=len(x)//f*f; return x[:n].reshape(-1,f,x.shape[1]).mean(1)
def feats(acc,gyr):
    """acc (T,3) g, gyr (T,3) deg/s at 100Hz, T==W -> (3,W)"""
    an=np.linalg.norm(acc,axis=1); gn=np.linalg.norm(gyr,axis=1)/100.
    k=25; c=np.cumsum(np.vstack([np.zeros((1,3)),acc]),0); i=np.arange(len(acc)); j=np.maximum(0,i-k+1)
    lp=(c[i+1]-c[j])/(i+1-j)[:,None]   # causal moving average 0.25s
    ref=acc[:50].mean(0); ref/=np.linalg.norm(ref)+1e-9
    tilt=np.arccos(np.clip((lp@ref)/(np.linalg.norm(lp,axis=1)+1e-9),-1,1))
    return np.stack([an,gn,tilt]).astype(np.float32)

def windows_file(x):
    if len(x)<W: x=np.vstack([np.repeat(x[:1],W-len(x),0),x])
    return np.stack([feats(x[s:s+W,:3],x[s:s+W,3:]) for s in range(0,len(x)-W+1,STRIDE)])

# ---------- synthetic windows ----------
def synth_windows(rng,path,per=4):
    z=np.load(path,allow_pickle=True); X,y,imp,kind=z["X"],z["y"],z["imp"],z["kind"]
    Xs=[];ys=[];gs=[]
    for i,(x,lab,im) in enumerate(zip(X,y,imp)):
        x=down(np.asarray(x,np.float32),5)   # 500->100Hz
        L=len(x)
        for _ in range(per):
            if lab==1:
                ii=im//5; lo=max(0,ii-250); hi=min(L-W,ii-150)
                if hi<lo: continue
                s=rng.integers(lo,hi+1)
            else: s=rng.integers(0,L-W+1)
            w=x[s:s+W].copy(); g=rng.uniform(0.95,1.05)
            w[:,:3]=w[:,:3]*g+rng.normal(0,0.02,(W,3))+rng.normal(0,0.02,3); w[:,3:]+=rng.normal(0,0.5,(W,3))
            w[:,:3]=np.clip(w[:,:3],-16,16)
            Xs.append(feats(w[:,:3],w[:,3:])); ys.append(lab); gs.append(i)
    return np.stack(Xs),np.array(ys,np.float32),np.array(gs),kind

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(3,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
                          nn.Conv1d(64,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU())
        s.h=nn.Linear(128,1)
    def forward(s,x): z=s.f(x); return s.h(torch.cat([z.mean(2),z.amax(2)],1)).squeeze(1)

def train(X,y,epochs=15,tag=""):
    net=Net().to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3); pw=torch.tensor((y==0).sum()/max(1,(y==1).sum()),device=DEV)
    lossf=nn.BCEWithLogitsLoss(pos_weight=pw); Xt=torch.tensor(X,device=DEV); yt=torch.tensor(y,device=DEV)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Xt),device=DEV); tot=0
        for b in range(0,len(perm),128):
            idx=perm[b:b+128]; opt.zero_grad(); l=lossf(net(Xt[idx]),yt[idx]); l.backward(); opt.step(); tot+=l.item()*len(idx)
        if ep%5==4 or ep==epochs-1: print(f"  [{tag}] ep{ep+1} loss {tot/len(Xt):.4f}",flush=True)
    net.eval(); return net

@torch.no_grad()
def predict(net,X): return torch.cat([torch.sigmoid(net(torch.tensor(X[i:i+4096],device=DEV))) for i in range(0,len(X),4096)]).cpu().numpy()

def auc(pos,neg):
    pos=np.asarray(pos);neg=np.asarray(neg); return float(((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean()))
