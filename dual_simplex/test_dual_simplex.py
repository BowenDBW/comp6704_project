"""Correctness and failure-path tests. HiGHS appears only as an external oracle."""
import unittest
import numpy as np
from scipy.optimize import linprog
from dual_simplex import solve,Options


def model(cost,s,d,u):
    m,n=cost.shape; k=m*n
    A=np.block([[np.kron(np.eye(m),np.ones((1,n))),np.eye(m),np.zeros((m,k))],
                [np.tile(np.eye(n),(1,m)),np.zeros((n,m+k))],
                [np.eye(k),np.zeros((k,m)),np.eye(k)]])
    return A,np.r_[s,d,u.ravel()],np.r_[cost.ravel(),np.zeros(m+k)]


class Tests(unittest.TestCase):
    def certify(self,A,b,c,r):
        self.assertEqual(r['status'],'optimal',r['message'])
        self.assertLessEqual(np.max(np.abs(A@r['z']-b)),1e-7)
        self.assertGreaterEqual(r['z'].min(),-1e-8)
        self.assertGreaterEqual((c-A.T@r['y']).min(),-1e-7)
        self.assertLessEqual(abs(c@r['z']-b@r['y'])/max(1,abs(c@r['z']),abs(b@r['y'])),1e-7)

    def test_hand_computed_capacity_pivot(self):
        A,b,c=model(np.array([[1.],[3.]]),[2,2],[2],np.array([[1.],[2.]]))
        ans=solve(A,b,c); self.certify(A,b,c,ans)
        self.assertAlmostEqual(ans['objective'],4)
        self.assertGreater(ans['iterations'],0)

    def test_tied_costs_degeneracy_and_unchanged_warm(self):
        A,b,c=model(np.ones((2,2)),[1,1],[1,1],np.full((2,2),.5))
        ans=solve(A,b,c); self.certify(A,b,c,ans)
        again=solve(A,b,c,warm_start={'basis':ans['basis']})
        self.certify(A,b,c,again); self.assertEqual(again['iterations'],0)

    def test_random_small_and_warm_external_oracle(self):
        rng=np.random.default_rng(173)
        for _ in range(15):
            cost=rng.integers(1,6,(3,4)).astype(float); x=rng.uniform(1,3,(3,4))
            A,b,c=model(cost,1.2*x.sum(1),x.sum(0),1.3*x)
            copies=[v.copy() for v in (A,b,c)]
            base=solve(A,b,c); self.certify(A,b,c,base)
            bb=b.copy(); bb[3:7]*=rng.uniform(.95,1.05,4)
            saved=base['basis'].copy()
            for ans in [solve(A,bb,c),solve(A,bb,c,warm_start={'basis':saved})]:
                self.certify(A,bb,c,ans)
                oracle=linprog(c,A_eq=A,b_eq=bb,bounds=(0,None),method='highs')
                self.assertTrue(oracle.success)
                self.assertAlmostEqual(ans['objective'],oracle.fun,places=7)
            np.testing.assert_array_equal(saved,base['basis'])
            for got,before in zip((A,b,c),copies): np.testing.assert_array_equal(got,before)

    def test_limits_and_bad_inputs(self):
        A,b,c=model(np.array([[1.],[3.]]),[2,2],[2],np.array([[1.],[2.]]))
        self.assertEqual(solve(A,b,c,Options(max_iterations=0))['status'],'iteration_limit')
        self.assertEqual(solve(A,b,c,Options(time_limit=1e-15))['status'],'time_limit')
        with self.assertRaises(ValueError): solve(A,b,c,warm_start={'basis':np.zeros(len(b),dtype=int)})
        with self.assertRaises(ValueError): solve(A,b*np.nan,c)
        with self.assertRaises(ValueError): solve(A[:,::-1],b,c)

    def test_infeasible_not_reported_optimal(self):
        A,b,c=model(np.array([[1.]]),[1],[2],np.array([[1.]]))
        self.assertEqual(solve(A,b,c)['status'],'numerical_failure')

    def test_non_dual_feasible_warm_rejected(self):
        A,b,c=model(np.array([[1.],[3.]]),[2,2],[1],np.array([[2.],[2.]]))
        # Supply slack columns 2,3; expensive x column 1; capacity slack 4,5.
        r=solve(A,b,c,warm_start={'basis':np.array([2,3,1,4,5])})
        self.assertEqual(r['status'],'numerical_failure')


if __name__=='__main__': unittest.main(verbosity=2)
