"""Read-only benchmark runner. References are used only after solve returns."""
import argparse,csv,hashlib,json,platform,sys
from pathlib import Path
from statistics import median
import numpy as np
import scipy
from threadpoolctl import threadpool_limits,threadpool_info
from dual_simplex import solve,Options


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=Path('results')); ap.add_argument('--small-only',action='store_true')
    args=ap.parse_args(); root=args.benchmark; out=args.output
    if out.exists(): raise RuntimeError('Use a fresh output directory to preserve prior results')
    out.mkdir(parents=True); (out/'solutions').mkdir()
    before={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    ref={r['case_id']:float(r['highs_objective']) for r in csv.DictReader((root/'VALIDATION_SUMMARY.csv').open())}
    manifest=list(csv.DictReader((root/'benchmark_v1/manifest.csv').open()))
    bases=[r for r in manifest if r['kind']=='base']
    if args.small_only: bases=[r for r in bases if r['case_id']=='M03_N05_a030_seed2026_base']
    opt=Options(); rows=[]; summary=[]; counts={'failed':0,'warmups_failed':0}
    schema=next(csv.reader((root/'result_schema.csv').open()))
    def assess(ans,A,b,c,reference):
        if ans.get('z') is None or ans.get('y') is None: return False,None
        z,y,r=ans['z'],ans['y'],ans['r']
        f=float(c@z); d=float(b@y); err=abs(f-reference)/max(1,abs(reference))
        ok=(ans['status']=='optimal' and all(np.isfinite(t).all() for t in [z,y,r])
            and np.max(np.abs(A@z-b))<=1e-7 and z.min()>=-1e-8
            and np.max(np.abs(A.T@y+r-c))<=1e-7 and r.min()>=-1e-7
            and abs(f-d)/max(1,abs(f),abs(d))<=1e-7 and err<=1e-7)
        return bool(ok),err
    def batch(case,A,b,c,mode,warm=None,base_time=None):
        # warm state is copied inside the timed solver on every invocation.
        pre=solve(A,b,c,opt,warm); passed,_=assess(pre,A,b,c,ref[case])
        counts['warmups_failed']+=int(not passed)
        timings=[]; latest=None; success=0
        snapshot=None if warm is None else warm['basis'].copy()
        for rep in range(1,11):
            ans=solve(A,b,c,opt,warm); ok,err=assess(ans,A,b,c,ref[case]); latest=ans
            success+=int(ok); counts['failed']+=int(not ok)
            if ok: timings.append(ans['runtime_seconds'])
            row={k:ans.get(k,'') for k in schema}
            row.update(protocol_version='v1',case_id=case,algorithm='Dual Simplex',mode=mode,repeat_id=rep,
                       relative_objective_error=err,base_runtime_seconds='' if base_time is None else base_time,
                       environment_id='local_validation_threads1',options_id='dual_v1_defaults')
            row['validation_pass']=ok; row['initial_min_basic']=ans['initial_min_basic']
            rows.append(row)
            if ans['z'] is not None:
                name=f'{case}_{mode}_{rep:02d}'
                np.savez_compressed(out/'solutions'/f'{name}.npz',z=ans['z'],y=ans['y'],r=ans['r'],basis=ans['basis'])
                (out/'solutions'/f'{name}.json').write_text(json.dumps({k:v for k,v in ans.items() if k not in ['z','y','r','basis']},indent=2))
        if warm is not None and not np.array_equal(snapshot,warm['basis']): raise RuntimeError('Warm state mutated')
        summary.append(dict(case_id=case,mode=mode,successful_runs=success,total_runs=10,
                            median_runtime_seconds=median(timings) if timings else None,
                            iterations=latest['iterations'],base_runtime_seconds=base_time,
                            initial_min_basic=latest['initial_min_basic']))
        return latest
    with threadpool_limits(limits=1):
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                         platform=platform.platform(),processor=platform.processor(),threadpools=threadpool_info(),
                         scope='local validation, not final group timing comparison',options=opt.__dict__,
                         run_order='manifest base order; base cold, perturbed cold, perturbed warm; 1 warmup then 10 repeats',
                         factorization_reuse=False,algorithm_rng='none')
        for entry in bases:
            case=entry['case_id']; pert=case[:-4]+'demand_perturbed'
            with np.load(root/'benchmark_v1'/entry['npz_file']) as f: A,b,c=(f[k].copy() for k in ['A','b','c_hat'])
            with np.load(root/'benchmark_v1/instances'/f'{pert}.npz') as f:
                bb=f['b'].copy()
                if not np.array_equal(A,f['A']) or not np.array_equal(c,f['c_hat']): raise RuntimeError('Pair mismatch')
            base=batch(case,A,b,c,'cold'); batch(pert,A,bb,c,'cold')
            if base['status']!='optimal': raise RuntimeError('Cannot warm start from failed base')
            batch(pert,A,bb,c,'warm',{'basis':base['basis']},base['runtime_seconds'])
            print(case, 'completed',flush=True)
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    after={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}
    if before!=after: raise RuntimeError('Shared release changed')
    report=dict(cases=len(bases)*2,case_modes=len(summary),timed_runs=len(rows),warmup_runs=len(summary),**counts,
                all_pass=counts['failed']==0 and counts['warmups_failed']==0,shared_files_unchanged=len(before),
                max_primal_residual=max(float(r['primal_residual_inf']) for r in rows),
                max_relative_objective_error=max(float(r['relative_objective_error']) for r in rows),
                max_relative_duality_gap=max(float(r['relative_duality_gap']) for r in rows),
                zero_pivot_warm_cases=sum(s['mode']=='warm' and s['iterations']==0 for s in summary))
    (out/'validation.json').write_text(json.dumps(report,indent=2)); (out/'environment.json').write_text(json.dumps(environment,indent=2))
    (out/'input_sha256.json').write_text(json.dumps(before,indent=2))
    print(json.dumps(report,indent=2))
    if not report['all_pass']: raise RuntimeError('Validation failures recorded')


if __name__=='__main__': main()
