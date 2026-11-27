"""신뢰하는 자체 생성 NPZ로 원 합성 분할/동결 가중치의 검증 결과를 재계산."""
import argparse
import json
import numpy as np
from eval_sisfall import load
from train_eval import synth_windows, predict
from train_synth import split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('synth_npz'); args=parser.parse_args()
    rng=np.random.default_rng(0)
    x,y,g,_=synth_windows(rng,args.synth_npz); va=split(g,rng)
    out={}
    for name,m in [('train',~va),('val',va)]:
        out[name]={'simulations':int(len(np.unique(g[m]))),'positive_windows':int(y[m].sum()),'negative_windows':int((1-y[m]).sum())}
    y=y[va]; hit=predict(load(),x[va])>.5
    out['val'].update(tp=int(hit[y==1].sum()),fp=int(hit[y==0].sum()))
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
