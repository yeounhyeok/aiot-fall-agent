"""합성 NPZ만 받는 원 학습 레시피. 평가와 별도로 명시적으로 실행한다."""
import argparse
from pathlib import Path
import numpy as np
import torch
from train_eval import synth_windows, train, predict


def split(gs,rng):
    ids=np.unique(gs)
    return np.isin(gs,rng.choice(ids,len(ids)//5,replace=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('synth_npz',type=Path,help='직접 생성한 synth_v1.npz만 사용')
    ap.add_argument('--out',type=Path,required=True,help='새 체크포인트 파일')
    args=ap.parse_args()
    if args.out.exists(): ap.error('출력 체크포인트가 이미 존재합니다')
    rng=np.random.default_rng(0); torch.manual_seed(0)
    X,y,groups,_=synth_windows(rng,args.synth_npz)
    val=split(groups,rng)
    net=train(X[~val],y[~val],tag='synth')
    p=predict(net,X[val]); print('합성 검증 recall / FPR',*[float(np.mean(p[y[val]==i]>.5)) for i in (1,0)])
    with args.out.open('xb') as f: torch.save(net.state_dict(),f)

if __name__=='__main__':
    main()
