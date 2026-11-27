"""Focused numerical regression tests run by run_stage3.py in-process."""
import unittest
import numpy as np
from scipy.stats import qmc
from run_stage3 import bh_adjust, coeff_activation, jansen_total, select_top_tokens

class Stage3HelperTests(unittest.TestCase):
    def test_top_decile_stable_ties_and_size(self):
        x=np.array([[1.,4.,4.],[0.,3.,2.]])
        self.assertEqual(select_top_tokens(x,.5).tolist(),[1,2,4])
        self.assertEqual(select_top_tokens(x,.5).tolist(),select_top_tokens(x,.5).tolist())

    def test_coefficient_basis_reconstruction(self):
        u=np.array([[[1.,2.],[3.,4.]]]);w=np.array([[1.,0.,2.],[0.,2.,1.]])
        got=coeff_activation(u,w)
        self.assertTrue(np.array_equal(got,np.einsum('...k,kf->...f',u,w)))
        self.assertTrue(np.all(got>=0))

    def test_bh_adjustment_monotonicity(self):
        p=np.array([.04,.001,.03,.2]);q=bh_adjust(p);order=np.argsort(p)
        self.assertTrue(np.all(np.diff(q[order])>=-1e-12))
        self.assertTrue(np.all(q>=p))

    def test_jansen_known_additive_synthetic_indices(self):
        design=qmc.Sobol(8,scramble=True,seed=50991).random_base2(7)
        a,b=design[:,:4],design[:,4:]
        f=lambda z:z[:,0]+2*z[:,1]+3*z[:,2]+4*z[:,3]
        ya,yb=f(a),f(b);hy=[]
        for k in range(4):
            ab=a.copy();ab[:,k]=b[:,k];hy.append(f(ab))
        estimate,_=jansen_total(ya,yb,np.stack(hy));known=np.array([1,4,9,16])/30
        self.assertLessEqual(float(np.max(np.abs(estimate-known))),.10)

if __name__=='__main__':unittest.main()
