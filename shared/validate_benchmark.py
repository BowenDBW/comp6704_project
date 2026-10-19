"""Independent saved-file validation. Usage: python validate_benchmark.py benchmark_v1 SECOND_GENERATION"""
import csv, hashlib, itertools, json, platform, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import linprog

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def validate(first, second, destination):
    rows=list(csv.DictReader((first/'manifest.csv').open(encoding='utf-8')))
    expected={(m,n,a,seed,k) for (m,n),a,seed,k in itertools.product([(3,5),(5,10),(10,20)],[.3,.6,.9],[2026,2027,2028],['base','demand_perturbed'])}
    require(len(rows)==54,'case count')
    require({(int(r['M']),int(r['N']),float(r['alpha']),int(r['seed']),r['kind']) for r in rows}==expected,'case grid')
    config=json.loads((first/'benchmark_config.json').read_text())
    require(config==json.loads((second/'benchmark_config.json').read_text()),'config reproducibility')
    require(config['sizes_M_N']==[[3,5],[5,10],[10,20]] and config['alphas']==[.3,.6,.9] and config['seeds']==[2026,2027,2028] and config['highs_validated'],'config parameters')
    results=[]; cache={}; byte_equal={'npz':0,'mat':0}
    for r in rows:
        ident=r['case_id']; m,n,seed=int(r['M']),int(r['N']),int(r['seed']); a=float(r['alpha']); mn=m*n
        with np.load(first/r['npz_file'],allow_pickle=False) as f:
            require(set(f.files)=={'A','b','c_hat'},ident+' NPZ keys')
            A,b,c=(f[k].copy() for k in ['A','b','c_hat'])
        require(A.shape==(m+n+mn,2*mn+m) and b.shape==(m+n+mn,) and c.shape==(2*mn+m,),ident+' shapes')
        require(int(r['rows_A'])==A.shape[0] and int(r['columns_A'])==A.shape[1],ident+' manifest dimensions')
        require(all(np.isfinite(v).all() for v in [A,b,c]),ident+' finite input')
        S=np.kron(np.eye(m),np.ones((1,n))); D=np.tile(np.eye(n),(1,m))
        E=np.block([[S,np.eye(m),np.zeros((m,mn))],[D,np.zeros((n,m)),np.zeros((n,mn))],[np.eye(mn),np.zeros((mn,m)),np.eye(mn)]])
        require(np.array_equal(A,E),ident+' independent standard form')
        rng=np.random.default_rng(np.random.SeedSequence([seed,m,n,0]))
        costs=rng.uniform(1,20,(m,n)); demand=rng.uniform(10,30,n); w=rng.uniform(.5,1.5,(m,n))
        x=w/w.sum(axis=0)*demand; supply=1.2*x.sum(axis=1); cap=(1+2*(1-a))*x
        delta=np.random.default_rng(np.random.SeedSequence([seed,m,n,1])).uniform(-.05,.05,n)
        d=demand if r['kind']=='base' else demand*(1+delta)
        require(np.array_equal(b,np.concatenate([supply,d,cap.ravel()])),ident+' RHS construction')
        require(np.array_equal(c,np.r_[costs.ravel(),np.zeros(m+mn)]),ident+' objective construction')
        xx=x if r['kind']=='base' else x*(d/demand)
        z=np.r_[xx.ravel(),supply-xx.sum(axis=1),(cap-xx).ravel()]
        wr=float(np.max(np.abs(A@z-b))); require(wr<=1e-8 and z.min()>=-1e-8,ident+' witness')
        sol=linprog(c,A_eq=A,b_eq=b,bounds=(0,None),method='highs')
        require(sol.success and sol.status==0,ident+' HiGHS success')
        require(np.isfinite(sol.x).all() and np.isfinite(sol.fun),ident+' finite solution')
        for section in ['eqlin','ineqlin','lower','upper']:
            require(np.isfinite(sol[section].marginals).all(),ident+' finite multipliers')
        # Upper-bound residuals are +inf by design for unbounded-above variables.
        hr=float(np.max(np.abs(A@sol.x-b))); require(hr<1e-7 and sol.x.min()>=-1e-8,ident+' solution feasibility')
        dual=c-A.T@sol.eqlin.marginals
        dr=float(max(0,-dual.min())); gap=float(abs(c@sol.x-b@sol.eqlin.marginals))
        require(dr<=1e-7 and gap<=1e-7,ident+' dual certificate')
        original=linprog(costs.ravel(),A_ub=np.r_[S,np.eye(mn)],b_ub=np.r_[supply,cap.ravel()],A_eq=D,b_eq=d,bounds=(0,None),method='highs')
        require(original.success and abs(original.fun-sol.fun)<=1e-7,ident+' original model optimum')
        rounding=abs(float(r['highs_objective'])-sol.fun)
        require(np.isfinite(float(r['highs_objective'])) and rounding<=5e-8,ident+' manifest objective')
        require(digest(first/r['npz_file'])==r['sha256_npz'],ident+' manifest checksum')
        for suffix,key in [('npz','npz_file'),('mat','mat_file')]:
            p=first/r[key]; q=second/r[key]
            byte_equal[suffix]+=int(digest(p)==digest(q))
            for source in [p,q]:
                if suffix=='mat':
                    saved=loadmat(source)
                    require({k for k in saved if not k.startswith('__')}=={'A','b','c_hat'},ident+' MAT keys')
                    require(saved['b'].shape==(len(b),1) and saved['c_hat'].shape==(len(c),1),ident+' MAT columns')
                else:
                    with np.load(source,allow_pickle=False) as f: saved={k:f[k] for k in f.files}
                require(all(np.array_equal(saved[k].reshape(target.shape),target) for k,target in [('A',A),('b',b),('c_hat',c)]),ident+' consistency/reproducibility')
        cache[(m,n,seed,a,r['kind'])]=(A,b,c)
        results.append(dict(case_id=ident,M=m,N=n,alpha=a,seed=seed,kind=r['kind'],rows_A=A.shape[0],columns_A=A.shape[1],status='PASS',witness_residual_inf=wr,witness_min_z=float(z.min()),highs_status=sol.status,highs_objective=float(sol.fun),highs_residual_inf=hr,highs_min_z=float(sol.x.min()),dual_violation=dr,absolute_duality_gap=gap,original_model_objective=float(original.fun),manifest_rounding_error=rounding,npz_mat_exact=True,regeneration_exact=True,npz_sha256=digest(first/r['npz_file']),mat_sha256=digest(first/r['mat_file'])))
    for m,n,seed in itertools.product([3,5,10],[5,10,20],[2026,2027,2028]):
        if (m,n) not in [(3,5),(5,10),(10,20)]: continue
        for a in [.3,.6,.9]:
            A,b,c=cache[m,n,seed,a,'base']; AA,bb,cc=cache[m,n,seed,a,'demand_perturbed']
            require(np.array_equal(A,AA) and np.array_equal(c,cc) and np.array_equal(b[:m],bb[:m]) and np.array_equal(b[m+n:],bb[m+n:]),'pair invariants')
        for kind in ['base','demand_perturbed']:
            for low,high in [(.3,.6),(.6,.9)]:
                A,b,c=cache[m,n,seed,low,kind]; AA,bb,cc=cache[m,n,seed,high,kind]
                require(np.array_equal(A,AA) and np.array_equal(c,cc) and np.array_equal(b[:m+n],bb[:m+n]) and np.all(bb[m+n:]<b[m+n:]),'alpha invariants')
    with (destination/'VALIDATION_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=results[0]); writer.writeheader(); writer.writerows(results)
    stats=dict(status='PASS',instances=54,base=27,perturbed=27,npz_mat_exact=54,regeneration_exact=54,paired_checks=27,alpha_adjacent_checks=36,byte_identical_files=byte_equal,python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),rng=type(np.random.default_rng().bit_generator).__name__)
    for key in ['witness_residual_inf','highs_residual_inf','dual_violation','absolute_duality_gap','manifest_rounding_error']:
        stats['max_'+key]=max(r[key] for r in results)
    stats['objective_range']=[min(r['highs_objective'] for r in results),max(r['highs_objective'] for r in results)]
    (destination/'VALIDATION_RESULTS.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__':
    validate(Path(sys.argv[1]),Path(sys.argv[2]),Path(__file__).resolve().parent)
