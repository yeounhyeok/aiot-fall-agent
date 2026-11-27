import hashlib
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_eval import feats, windows_file, predict
from train_synth import split
from eval_sisfall import convert, load, metrics, WEIGHTS


class Detector(unittest.TestCase):
    def test_units(self):
        raw=np.zeros((5,9)); raw[:,0]=[256,256,0,512,0]; raw[:,3]=16384; raw[:,6:]=1000
        x=convert(raw)
        np.testing.assert_array_equal(x,np.tile([1,0,0,1000,0,0],(2,1)))

    def test_features_and_rotation(self):
        acc=np.zeros((300,3)); acc[:100,2]=1; acc[100:,0]=1
        w=np.zeros_like(acc); w[:,1]=200
        f=feats(acc,w); self.assertEqual(f.shape,(3,300)); np.testing.assert_allclose(f[0],1); np.testing.assert_allclose(f[1],2)
        self.assertLess(f[2,99],.001); self.assertAlmostEqual(f[2,124],np.pi/2,places=5)
        np.testing.assert_allclose(f,feats(acc[:,[1,2,0]],w[:,[1,2,0]]),atol=1e-5)

    def test_windows(self):
        x=np.tile([0.,0.,1.,0.,0.,0.],(325,1)); self.assertEqual(windows_file(x).shape,(2,3,300)); np.testing.assert_array_equal(windows_file(x[:8])[0],windows_file(x)[0]);self.assertEqual(windows_file(x[:324]).shape,(1,3,300))

    def test_split(self):
        groups=np.repeat(np.arange(25),4); mask=split(groups,np.random.default_rng(0));self.assertEqual(mask.sum(),20); self.assertFalse(set(groups[mask]) & set(groups[~mask]))

    def test_checkpoint(self):
        self.assertEqual(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),'65e192e9b62cbd7458219029458fe7a2eb9811232def307d637bbe7402c592e4');net=load();self.assertEqual(sum(p.numel() for p in net.parameters()),32001)
        x=np.tile([0.,0.,1.,0.,0.,0.],(325,1));p=predict(net,windows_file(x));self.assertEqual(p.shape,(2,));self.assertTrue(np.all((p>=0)&(p<.5)))

    def test_threshold_and_ties(self):
        m=metrics({'F1':.5,'F2':.8,'D1':.5,'D2':.55});self.assertEqual([m[k] for k in ['tp','fn','fp','tn','auc']],[1,1,1,1,.625])

    @unittest.skipUnless(os.environ.get('HUMANOID_XML') or (ROOT/'assets/humanoid.xml').exists(),'선택: 공식 XML 필요')
    def test_generator(self):
        from synth_v1 import run,FALLS,ADLS
        for i,kind in enumerate(FALLS+ADLS):
            with self.subTest(kind=kind):
                x,y,impact,_=run(kind,i*100000); self.assertEqual(x.shape,(2100,6));self.assertTrue(np.isfinite(x).all());self.assertEqual(y,int(i<3));self.assertEqual(impact,int(np.linalg.norm(x[:,:3],axis=1).argmax()) if y else -1)

if __name__=='__main__': unittest.main()
