"""Dual simplex for the shared transportation standard form; no external LP solver.

Basis indices are zero based. scipy.linalg is used only for LU linear solves.
Cold initialization is model-specific; pivots operate on the full supplied LP.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve, LinAlgWarning


@dataclass(frozen=True)
class Options:
    primal_tol: float = 1e-7
    nonnegative_tol: float = 1e-8
    dual_tol: float = 1e-7
    gap_tol: float = 1e-7
    pivot_tol: float = 1e-12
    internal_dual_tol: float = 1e-9
    max_iterations: int = 10000
    time_limit: float = 120.0


def transportation_shape(A, c):
    """Recognize exactly the documented matrix, not arbitrary equality LPs."""
    L,V=A.shape
    for m in range(1,L):
        if (V-m)%2: continue
        k=(V-m)//2
        if k<=0 or k%m: continue
        n=k//m
        if m+n+k!=L: continue
        S=np.kron(np.eye(m),np.ones((1,n)))
        D=np.tile(np.eye(n),(1,m))
        expected=np.block([[S,np.eye(m),np.zeros((m,k))],
                           [D,np.zeros((n,m+k))],
                           [np.eye(k),np.zeros((k,m)),np.eye(k)]])
        if np.array_equal(A,expected) and np.all(c[k:]==0): return m,n
    raise ValueError('Cold start requires the exact shared transportation matrix and zero slack costs')


def solve(A,b,c_hat,options=None,warm_start=None):
    """Return z,y,r,basis,metrics,history,status; never receives reference values.

    warm_start is {'basis': integer array}; only basis indices are reused.
    Input validation, copies, initialization and all factorizations are timed.
    Invalid inputs raise ValueError; numerical/iteration/time outcomes return status.
    """
    start=perf_counter()
    opt=options or Options()
    if not isinstance(opt,Options): raise ValueError('options must be Options')
    if opt.max_iterations<0 or opt.time_limit<=0: raise ValueError('Invalid resource limit')
    if any(not np.isfinite(v) or v<=0 for k,v in asdict(opt).items() if k!='max_iterations'):
        raise ValueError('Tolerances and time limit must be positive finite')
    A=np.array(A,dtype=np.float64,copy=True); b=np.array(b,dtype=np.float64,copy=True).reshape(-1)
    c=np.array(c_hat,dtype=np.float64,copy=True).reshape(-1)
    if A.ndim!=2 or A.shape!=(len(b),len(c)) or not all(np.isfinite(x).all() for x in [A,b,c]):
        raise ValueError('Incompatible shapes or nonfinite input')
    L,V=A.shape
    if warm_start is None:
        m,n=transportation_shape(A,c); k=m*n
        cheapest=np.argmin(c[:k].reshape(m,n),axis=0)*n+np.arange(n)
        basis=np.r_[np.arange(k,k+m),cheapest,np.arange(k+m,V)].astype(int)
        initialization='cheapest route per customer plus all slacks'
    else:
        raw=np.asarray(warm_start['basis'])
        if raw.shape!=(L,) or not np.issubdtype(raw.dtype,np.integer): raise ValueError('Basis must be L integer indices')
        basis=raw.astype(int,copy=True)
        if len(np.unique(basis))!=L or np.any(basis<0) or np.any(basis>=V): raise ValueError('Invalid basis indices')
        initialization='copied basis indices only; no factorization reused'
    history=[]; pivots=0; status='numerical_failure'; message=''; z=y=r=None
    initial_min_basic=None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',LinAlgWarning)
            while True:
                B=A[:,basis]; lu=lu_factor(B,check_finite=False)
                xb=lu_solve(lu,b,check_finite=False)
                y=lu_solve(lu,c[basis],trans=1,check_finite=False)
                r=c-A.T@y
                z=np.zeros(V); z[basis]=xb
                if not all(np.isfinite(x).all() for x in [xb,y,r]): raise FloatingPointError('Nonfinite linear solve')
                if initial_min_basic is None: initial_min_basic=float(xb.min())
                f=float(c@z); dy=float(b@y)
                metrics=dict(objective=f,primal_residual_inf=float(np.max(np.abs(A@z-b))),
                             min_z=float(z.min()),dual_residual_inf=float(np.max(np.abs(A.T@y+r-c))),
                             min_dual_slack=float(r.min()),relative_duality_gap=abs(f-dy)/max(1,abs(f),abs(dy)))
                history.append(dict(iteration=pivots,**metrics))
                if r.min() < -opt.internal_dual_tol:
                    message='Basis not dual feasible at internal tolerance'; break
                if (metrics['primal_residual_inf']<=opt.primal_tol and metrics['min_z']>=-opt.nonnegative_tol
                    and metrics['dual_residual_inf']<=opt.dual_tol and metrics['min_dual_slack']>=-opt.dual_tol
                    and metrics['relative_duality_gap']<=opt.gap_tol):
                    status='optimal'; message='Primal/dual feasibility and gap passed'; break
                if perf_counter()-start>=opt.time_limit:
                    status='time_limit'; message='Time budget reached at iteration boundary'; break
                if pivots>=opt.max_iterations:
                    status='iteration_limit'; message='Pivot budget reached'; break
                bad=np.flatnonzero(xb < -opt.nonnegative_tol)
                if not len(bad): message='No infeasible basic variable but common checks fail'; break
                # Dual Bland rule: smallest indexed infeasible basic variable.
                leaving=int(bad[np.argmin(basis[bad])])
                e=np.zeros(L); e[leaving]=1
                row=lu_solve(lu,e,trans=1,check_finite=False)@A
                nonbasic=np.ones(V,dtype=bool); nonbasic[basis]=False
                candidates=np.flatnonzero(nonbasic & (row < -opt.pivot_tol))
                if not len(candidates):
                    message='No eligible pivot at numerical threshold; no infeasibility claim made'; break
                # Minimum dual ratio; exact ties resolved by smallest column index.
                ratios=np.maximum(r[candidates],0)/(-row[candidates])
                entering=int(candidates[np.argmin(ratios)])
                history[-1].update(leaving_column=int(basis[leaving]),entering_column=entering,
                                   pivot=float(row[entering]))
                basis[leaving]=entering; pivots+=1
    except (np.linalg.LinAlgError,LinAlgWarning,FloatingPointError,ValueError) as exc:
        status='numerical_failure'; message=str(exc)
    result=dict(status=status,message=message,z=z,y=y,r=r,basis=basis.copy(),iterations=pivots,
                phase1_iterations=0,phase2_iterations=pivots,initialization=initialization,
                initial_min_basic=initial_min_basic,history=history,options=asdict(opt))
    if z is not None and y is not None and r is not None and all(np.isfinite(t).all() for t in [z,y,r]):
        f=float(c@z); d=float(b@y)
        result.update(objective=f,primal_residual_inf=float(np.max(np.abs(A@z-b))),min_z=float(z.min()),
                      dual_residual_inf=float(np.max(np.abs(A.T@y+r-c))),min_dual_slack=float(r.min()),
                      relative_duality_gap=abs(f-d)/max(1,abs(f),abs(d)))
    result['runtime_seconds']=perf_counter()-start
    return result
