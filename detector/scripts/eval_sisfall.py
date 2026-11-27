"""원시 SisFall 파일에서 바로 시험한다. 캐시/개별 예측을 저장하지 않는다."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_eval import DEV, Net, down, windows_file, predict, auc

WEIGHTS = Path(__file__).resolve().parents[1] / 'weights' / 'net_synth_v1.pt'
TEST = {f'SA{i:02}' for i in range(16, 24)} | {f'SE{i:02}' for i in range(1, 16)}


def convert(a):
    acc = a[:, :3] * (32 / 2**13)
    gyro = a[:, 3:6] * (4000 / 2**16)
    return down(np.concatenate([acc, gyro], axis=1), 2).astype(np.float32)


def read_record(path):
    with path.open() as f:
        rows = [[float(v) for v in line.strip().rstrip(';').split(',')]
                for line in f if line.strip()]
    a = np.asarray(rows)
    if a.ndim != 2 or a.shape[1] < 6 or len(a) < 2:
        raise ValueError(f'잘못된 SisFall 레코드: {path.name}')
    return convert(a)


def records(root):
    # 원 코드 res[파일명]=점수와 동일: 정렬 후 마지막 경로가 중복 ID를 대체한다.
    paths = {}
    for path in sorted(root.glob('S*/*.txt')):
        if path.stem.split('_')[1] in TEST:
            paths[path.stem] = path
    return paths


def load(weights=WEIGHTS):
    net = Net().to(DEV)
    net.load_state_dict(torch.load(weights, map_location=DEV, weights_only=True), strict=True)
    return net.eval()


def metrics(scores):
    pos, neg = [np.array([s for n,s in scores.items() if n.startswith(c)]) for c in 'FD']
    tp, fp = int((pos > .5).sum()), int((neg > .5).sum())
    if not (len(pos) and len(neg)):
        raise ValueError('두 클래스가 모두 필요합니다')
    return dict(n_fall=len(pos), n_adl=len(neg), tp=tp, fn=len(pos)-tp, fp=fp, tn=len(neg)-fp,
                recall=tp/len(pos), fpr=fp/len(neg), auc=auc(pos,neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sisfall', type=Path, help='압축을 푼 SisFall_dataset 폴더')
    ap.add_argument('--weights', type=Path, default=WEIGHTS)
    args=ap.parse_args()
    paths=records(args.sisfall)
    if (sum(n.startswith('F') for n in paths),sum(n.startswith('D') for n in paths)) != (673,1522):
        ap.error('원 시험셋과 ID 개수가 다릅니다. SisFall_dataset 경로와 배포판을 확인하세요.')
    net=load(args.weights)
    scores={n:float(predict(net,windows_file(read_record(p))).max()) for n,p in paths.items()}
    print(json.dumps(metrics(scores),indent=2))

if __name__=='__main__':
    main()
