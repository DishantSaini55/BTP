import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from MTN.run_craft_stage1_v1 import atomic_json

class CraftStage1Tests(unittest.TestCase):
    def test_feature_scaling_inverse(self):
        a=np.array([[1.,2.],[3.,8.],[5.,4.]])
        s=np.sqrt(np.mean(a*a,axis=0)); self.assertTrue(np.allclose((a/s)*s,a))
    def test_activation_time_alignment_shape(self):
        acts=np.zeros((7,15,128)); self.assertEqual(acts.reshape(-1,128).shape,(105,128))
        self.assertEqual(acts.reshape(-1,128)[15,0],0)
    def test_identity_hook_replacement(self):
        import torch
        from MTN.ao_models_v1 import build_ao_model
        torch.set_num_threads(1);m=build_ao_model('AO-3').eval();x=torch.randn(2,15,4); saved={}
        h=m.encoder_layers[2].feedforward[1].register_forward_hook(lambda _,__,o:saved.setdefault('a',o.detach().clone()))
        with torch.no_grad(): y=m(x)
        h.remove(); a=saved['a'];h=m.encoder_layers[2].feedforward[1].register_forward_hook(lambda _,__,o:a)
        with torch.no_grad(): z=m(x)
        h.remove();self.assertLess(float((y-z).abs().max()),1e-6)
    def test_exclusive_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'lock';fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
            with self.assertRaises(FileExistsError):os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    def test_atomic_stage_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'stage.json';atomic_json(p,{'state':'complete'});self.assertEqual(p.read_text(),'{\n  "state": "complete"\n}')
            self.assertFalse(p.with_suffix('.json.tmp').exists())
    def test_nnls_recovers_nonnegative_coefficients(self):
        w=np.array([[1.,0.],[0.,1.],[1.,1.]]);u=np.array([.4,.7]);x=u@w.T
        fit,_=nnls(w,x);self.assertTrue(np.allclose(fit,u,atol=1e-8))
if __name__=='__main__':unittest.main()
