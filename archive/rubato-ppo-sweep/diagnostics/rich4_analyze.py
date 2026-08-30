"""Dip anatomy: rejection ladders e(dt), floor-commits, solver convergence."""
import glob
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(BASE, "rich", "rich_*.npz")))
data = [np.load(p, allow_pickle=True) for p in parts]
att = np.concatenate([d["att"] for d in data])
rej = np.concatenate([d["rej"] for d in data])
fc = np.concatenate([d["fc"] for d in data])
niter = np.concatenate([d["niter"] for d in data])
dt = np.concatenate([d["dt"] for d in data])
ncon = np.concatenate([d["ncon"] for d in data])
mind = np.concatenate([d["mindist"] for d in data])
qid = sum([list(d["qid"]) for d in data], [])
tdt = sum([list(d["tdt"]) for d in data], [])
te = sum([list(d["te"]) for d in data], [])
F, N = att.shape
TOL = 1e-3
print(f"frames={F} worlds={N}")

print("\n== HEADLINE COUNTERS ==")
print(f"rejects: total={rej.sum()} = {rej.sum()/att.sum()*100:.1f}% of attempts; worlds-frames with >=1 reject: {(rej>0).mean()*100:.2f}%")
print(f"reject count dist (per world-frame, where >0): p50={np.percentile(rej[rej>0],50):.0f} p99={np.percentile(rej[rej>0],99):.0f} max={rej.max()}")
print(f"FLOOR-COMMITS (e>tol accepted at dt_min): total={fc.sum()} in {(fc>0).sum()} world-frames ({(fc>0).mean()*100:.4f}%)")
if (fc > 0).any():
    print(f"  floor-commit count where >0: p50={np.percentile(fc[fc>0],50):.0f} max={fc.max()}")
print(f"solver_niter (end-of-frame): p50={np.percentile(niter,50):.0f} p99={np.percentile(niter,99):.0f} max={niter.max()} | at-cap(>=100): {(niter>=100).mean()*100:.3f}%")

print("\n== REJECTION LADDERS (per-attempt e(dt) traces of selected worlds) ==")
# classify each rejection cascade: as dt shrinks 10x, does e fall ~100x (convergent, order 2) or stay flat?
ratios = []
deep_ladders = []
for f in range(F):
    ids = list(qid[f])
    for i, w in enumerate(ids):
        n = att[f, w]
        if n < 1 or n > 32:
            n = min(n, 32)
        ldt, le = tdt[f][i][:n], te[f][i][:n]
        rejmask = le > TOL
        if rejmask.sum() >= 2:
            # consecutive rejected attempts: e-ratio vs dt-ratio => local convergence order
            idx = np.flatnonzero(rejmask)
            for a, b in zip(idx[:-1], idx[1:]):
                if b == a + 1 and ldt[a] > 0 and ldt[b] > 0 and ldt[b] < ldt[a]:
                    p = np.log(le[a] / max(le[b], 1e-12)) / np.log(ldt[a] / ldt[b])
                    ratios.append((ldt[b], p))
        if rejmask.sum() >= 3 and ldt[rejmask].min() < 1e-4:
            deep_ladders.append((f, w, ldt.copy(), le.copy()))
ratios = np.array(ratios) if ratios else np.zeros((0, 2))
print(f"consecutive-rejection pairs: {len(ratios)}; observed local order p = log(e1/e2)/log(dt1/dt2):")
if len(ratios):
    for lo, hi in [(1e-3, 1e-2), (1e-4, 1e-3), (1e-5, 1e-4), (0, 1e-5)]:
        m = (ratios[:, 0] >= lo) & (ratios[:, 0] < hi)
        if m.sum():
            print(f"  dt in [{lo:.0e},{hi:.0e}): n={m.sum()} order p p50={np.percentile(ratios[m,1],50):.2f} "
                  f"p10={np.percentile(ratios[m,1],10):.2f} (p~2 = convergent truncation, p~0 = plateau/discontinuity)")
print(f"\nexemplar deep ladders (attempt: dt -> e), {min(4,len(deep_ladders))} of {len(deep_ladders)}:")
for f, w, ldt, le in deep_ladders[:4]:
    steps = " | ".join(f"{d:.1e}->{e:.1e}{'R' if e>TOL else 'A'}" for d, e in zip(ldt, le))
    print(f"  f={f} w={w}: {steps}")

print("\n== WHERE DO DEEP DIPS LIVE (contact state of ladder worlds with min rejected dt < 1e-4) ==")
deepset = set((f, w) for f, w, _, _ in deep_ladders)
if deep_ladders:
    dn = np.array([ncon[f, w] for f, w, _, _ in deep_ladders])
    dm = np.array([mind[f, w] for f, w, _, _ in deep_ladders])
    print(f"n={len(deep_ladders)}: in-contact {np.mean(dn>0)*100:.0f}%, ncon p50={np.percentile(dn,50):.0f}, "
          f"deepest-pen p50={np.nanpercentile(np.where(np.isfinite(dm),dm,np.nan),50)*1e3:.1f}mm")
    dni = np.array([niter[f, w] for f, w, _, _ in deep_ladders])
    print(f"solver_niter of dip worlds: p50={np.percentile(dni,50):.0f} p90={np.percentile(dni,90):.0f} at-cap: {(dni>=100).mean()*100:.1f}%")
