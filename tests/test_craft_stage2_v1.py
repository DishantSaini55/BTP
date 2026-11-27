import unittest
import numpy as np
from scipy.optimize import nnls

class CraftStage2Numerics(unittest.TestCase):
    def test_nnls_orientation_and_reconstruction(self):
        W=np.array([[1.,0.],[0.,1.],[1.,1.],[.5,.2]])
        U=np.array([[.3,.8],[1.2,.1]])
        X=U@W.T
        fit=np.vstack([nnls(W,row)[0] for row in X])
        self.assertTrue(np.allclose(fit,U,atol=1e-8))
        self.assertLess(np.linalg.norm(X-fit@W.T),1e-8)
    def test_rms_scale_apply_and_inverse(self):
        train=np.array([[1.,2.],[3.,6.],[2.,4.]])
        scale=np.sqrt(np.mean(train**2,axis=0));z=train/scale
        self.assertTrue(np.allclose(z*scale,train))
    def test_nearzero_feature_filter_is_train_derived(self):
        train=np.array([[1.,0.],[2.,0.]])
        rms=np.sqrt(np.mean(train**2,axis=0));keep=rms>rms.max()*1e-5
        self.assertEqual(keep.tolist(),[True,False])
if __name__=='__main__':unittest.main()
