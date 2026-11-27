"""MuJoCo synthetic IMU (pelvis) v1: standing-start falls + ADLs.
Output: data/synth_v1.npz  (X: object array of (T,6) [acc g, gyro deg/s] @500Hz, y, kind, impact_idx)
"""
import mujoco, numpy as np, re, os, sys, json
from multiprocessing import Pool
HERE=os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
import hashlib
asset = Path(os.environ.get("HUMANOID_XML", Path(HERE).parent / "assets" / "humanoid.xml"))
raw = asset.read_bytes()
if hashlib.sha256(raw).hexdigest() != "acc9167550bf20580717a551467e6e8d0d03d1916d24bbe4b734ce11afb99406":
    raise ValueError("humanoid.xml SHA-256 mismatch")
xml = raw.decode("utf-8")
xml=re.sub(r'(<body name="pelvis"[^>]*>)', r'\1<site name="imu" pos="0 0 0" size="0.02"/>', xml, count=1)
xml=xml.replace("</mujoco>",'<sensor><accelerometer name="acc" site="imu"/><gyro name="gyr" site="imu"/></sensor></mujoco>')
G=9.81; DT=0.002; REC0=0.8
FALLS=["topple","trip_like","limp"]
ADLS=["stand","stumble_recover","squat","bend","jump"]

def build(rng):
    m=mujoco.MjModel.from_xml_string(xml); m.opt.timestep=DT
    m.opt.enableflags|=mujoco.mjtEnableBit.mjENBL_OVERRIDE
    tc=rng.uniform(0.06,0.14); m.opt.o_solref[:]=[tc,1.0]; m.opt.o_solimp[:]=[0.9,0.95,0.001,0.5,2]; m.opt.o_margin=0
    d=mujoco.MjData(m); q0=m.qpos0.copy()
    ms=rng.uniform(0.8,1.25); m.body_mass[:]*=ms
    q=q0.copy(); q[7:]+=rng.normal(0,0.04,len(q)-7); d.qpos[:]=q; d.qpos[2]=1.45
    k=rng.uniform(300,500); kh=rng.uniform(2200,3000); m.jnt_stiffness[1:]=kh; m.jnt_stiffness[0]=0; m.qpos_spring[:]=q0; m.dof_damping[6:]=60.
    tid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"torso")
    return m,d,q0,k,kh,tid,dict(solref_tc=tc,mass_scale=ms,k_release=k,k_hold=kh)

def assist(m,d,tid,feet,Kp,Kd):
    # 무게중심(전신)을 두 발 중심 위로 되돌리는 수평 힘 = 발목 전략의 지면반력 대용
    com=d.subtree_com[1,:2]; fc=0.5*(d.xpos[feet[0],:2]+d.xpos[feet[1],:2])
    mass=m.body_subtreemass[1]
    mujoco.mj_subtreeVel(m,d); v=d.subtree_linvel[1,:2]
    d.xfrc_applied[tid,:2]+= -Kp*mass*(com-fc) - Kd*mass*v

def run(kind,seed,T=5.0):
    rng=np.random.default_rng(seed); m,d,q0,k,kh,tid,P=build(rng)
    sq=m.key_qpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"squat")].copy()
    bend=q0.copy()
    for j in ("abdomen_y","hip_y_right","hip_y_left"):
        a=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j)]; bend[a]=sq[a]
    feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ("foot_right","foot_left")]
    assert min(feet)>=0
    Kp=rng.uniform(40,80); Kd=rng.uniform(8,16)
    t0=rng.uniform(1.6,2.2); ang=rng.uniform(-np.pi,np.pi); rel=False
    if kind=="topple": tone=rng.uniform(0.02,0.4); F=rng.uniform(300,500); dur=rng.uniform(0.1,0.2)
    elif kind=="trip_like": ang=rng.uniform(-0.61,0.61); tone=rng.uniform(0.4,0.9); F=rng.uniform(300,500); dur=rng.uniform(0.1,0.2)
    elif kind=="limp": tone=rng.uniform(0,0.05); F=rng.uniform(60,200); dur=rng.uniform(0.1,0.3)
    elif kind=="stumble_recover": F=rng.uniform(60,220); dur=rng.uniform(0.1,0.2)
    elif kind=="jump": F=rng.uniform(900,1800); dur=rng.uniform(0.08,0.15)
    elif kind in("squat","bend"): down=rng.uniform(0.4,1.5); hold=rng.uniform(0.3,1.0); up=rng.uniform(0.4,1.5); depth=rng.uniform(0.5,1.0); tgt=sq if kind=="squat" else bend
    elif kind=="lie_slow": rdur=rng.uniform(0.8,1.8); damp=rng.uniform(40,140)
    fd=15.*rng.uniform(0.7,1.3)
    sway_f=rng.normal(0,8,2); A=[];Gy=[]
    for i in range(int(T/DT)):
        t=d.time; d.xfrc_applied[:]=0
        if not (kind in FALLS and t>=t0) and not (kind=='lie_slow' and t>=t0): assist(m,d,tid,feet,Kp,Kd)
        if kind in FALLS:
            if t0<=t<t0+dur: d.xfrc_applied[tid,:2]=F*np.array([np.cos(ang),np.sin(ang)])
            if t>=t0 and not rel: rel=True; m.jnt_stiffness[1:]=k*tone; m.dof_damping[6:]=fd
        elif kind=="stand":
            if i%250==0: sway_f=rng.normal(0,8,2)
            d.xfrc_applied[tid,:2]+=sway_f
        elif kind=="stumble_recover":
            if t0<=t<t0+dur: d.xfrc_applied[tid,:2]=F*np.array([np.cos(ang),np.sin(ang)])
        elif kind=="jump":
            if t0<=t<t0+dur: d.xfrc_applied[tid,2]=F
        elif kind in("squat","bend"):
            s=0.
            if t0<=t<t0+down: s=0.5-0.5*np.cos(np.pi*(t-t0)/down)
            elif t0+down<=t<t0+down+hold: s=1.
            elif t0+down+hold<=t<t0+down+hold+up: s=0.5+0.5*np.cos(np.pi*(t-t0-down-hold)/up)
            m.qpos_spring[7:]=q0[7:]+depth*s*(tgt[7:]-q0[7:])
        elif kind=="lie_slow":
            if t>=t0:
                s=min(1.,(t-t0)/rdur); m.jnt_stiffness[1:]=kh*(1-s)+1e-3; m.dof_damping[6:]=60+s*damp
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qacc).all(): return None
        if t>=REC0: A.append(d.sensordata[:3].copy()); Gy.append(d.sensordata[3:6].copy())
    A=np.array(A)/G; Gy=np.degrees(np.array(Gy)); z=float(d.xpos[tid,2]); n=np.linalg.norm(A,axis=1)
    n200=n[:len(n)//5*5].reshape(-1,5).mean(1)
    fell=z<0.45; up=z>1.0
    # validity: falls must fall; upright ADLs must stay up; lie_slow must be gentle
    if kind in FALLS and not fell: return None
    if kind in ("stand","stumble_recover","squat","bend","jump") and not up: return None
    if kind=="lie_slow" and (not fell or n200.max()>3.0): return None
    imp=int(n.argmax()) if kind in FALLS else -1
    P.update(kind=kind,seed=seed,end_z=z,peak200=float(n200.max()))
    return np.concatenate([A,Gy],1).astype(np.float32),int(kind in FALLS),imp,P

def job(a): return run(*a)
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="MuJoCo synth_v1 기록 생성")
    parser.add_argument("N",type=int,help="동작별 최대 기록 수 (원실험: 300)")
    parser.add_argument("--out",type=Path,required=True,help="새 출력 디렉터리")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    N=args.N; jobs=[(k,100000*ki+s) for ki,k in enumerate(FALLS+ADLS) for s in range(int(N*1.6))]
    with Pool(2) as p: R=p.map(job,jobs,chunksize=4)
    out={k:[] for k in FALLS+ADLS}
    for r in R:
        if r is not None and len(out[r[3]["kind"]])<N: out[r[3]["kind"]].append(r)
    X=[];y=[];imp=[];meta=[]
    for k,L in out.items():
        print(k,"kept",len(L),"of",sum(1 for j in jobs if j[0]==k),"| peak200 med %.1f"%np.median([r[3]["peak200"] for r in L]) if L else "")
        for x,lab,i,P in L: X.append(x);y.append(lab);imp.append(i);meta.append(P)
    Xo=np.empty(len(X),dtype=object); Xo[:]=X
    np.savez(args.out/"synth_v1.npz",X=Xo,y=np.array(y),imp=np.array(imp),kind=np.array([m["kind"] for m in meta]))
    json.dump(meta,open(args.out/"synth_v1_meta.json","w"))
