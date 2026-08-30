"""Final attribution: does velocity (spin/joint speed) explain the expensive worlds?"""
import glob
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
parts = sorted(glob.glob(os.path.join(BASE, "rich", "rich_*.npz")))
data = [np.load(p, allow_pickle=True) for p in parts]
att = np.concatenate([d["att"] for d in data])
ncon = np.concatenate([d["ncon"] for d in data])
mind = np.concatenate([d["mindist"] for d in data])
cw = np.concatenate([d["cube_w"] for d in data])    # cube |angular vel| rad/s
cv = np.concatenate([d["cube_v"] for d in data])    # cube |linear vel| m/s
jv = np.concatenate([d["joint_v"] for d in data])   # max |joint vel| rad/s
resets = sum([list(d["reset"]) for d in data], [])
F, N = att.shape
print(f"frames={F} worlds={N}")

print("\n== ATTEMPTS vs CUBE SPIN ==")
for lo, hi in [(0, 5), (5, 20), (20, 50), (50, 150), (150, 500), (500, 1e9)]:
    m = (cw >= lo) & (cw < hi)
    if m.sum():
        print(f"  |w_cube| [{lo:4.0f},{hi if hi<1e9 else 9999:4.0f}) rad/s: {m.mean()*100:6.2f}% of world-frames, "
              f"att mean={att[m].mean():5.2f} p99={np.percentile(att[m],99):4.0f} max={att[m].max():3.0f}")

print("\n== ATTEMPTS vs MAX JOINT SPEED ==")
for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)]:
    m = (jv >= lo) & (jv < hi)
    if m.sum():
        print(f"  |qd_joint| [{lo:3.0f},{hi if hi<1e9 else 999:3.0f}) rad/s: {m.mean()*100:6.2f}% of world-frames, "
              f"att mean={att[m].mean():5.2f} p99={np.percentile(att[m],99):4.0f}")

print("\n== WHAT ARE THE EXPENSIVE WORLDS (att >= 10)? ==")
exp = att >= 10
print(f"expensive world-frames: {exp.mean()*100:.2f}%")
for name, m in [("cube spinning >50 rad/s", cw > 50), ("cube spinning >150", cw > 150),
                ("fast joints >10 rad/s", jv > 10), ("in contact", ncon > 0),
                ("deep pen < -5mm", np.where(np.isfinite(mind), mind, 1) < -5e-3),
                ("cube fast linear >1 m/s", cv > 1.0)]:
    print(f"  P({name} | expensive) = {m[exp].mean()*100:5.1f}%   P(expensive | {name}) = {exp[m].mean()*100:5.1f}%  base P={m.mean()*100:5.1f}%")

print("\n== BINDING WORLD (frame argmax) velocity profile ==")
kmax = att.argmax(1)
bm = np.zeros_like(att, bool)
bm[np.arange(F), kmax] = True
for nm, arr in [("cube_w", cw), ("cube_v", cv), ("joint_v", jv)]:
    print(f"  {nm}: binding p50={np.percentile(arr[bm],50):7.1f} p90={np.percentile(arr[bm],90):7.1f}   all p50={np.percentile(arr,50):6.2f} p90={np.percentile(arr,90):6.2f}")

print("\n== QUIET-WORLD BASELINE (no contact, slow cube, slow joints) ==")
quiet = (ncon == 0) & (cw < 5) & (jv < 2)
print(f"quiet: {quiet.mean()*100:.2f}% of world-frames, att mean={att[quiet].mean():.2f} p99={np.percentile(att[quiet],99):.0f}")
active = ~quiet
print(f"non-quiet: att mean={att[active].mean():.2f}")

print("\n== TIME-SINCE-RESET vs depth & spin ==")
last = np.full(N, -(10**9))
tsr = np.zeros((F, N), np.int64)
for f in range(F):
    for w in resets[f]:
        last[int(w)] = f
    tsr[f] = f - last
mfin = np.where(np.isfinite(mind), mind, np.nan)
for lo, hi in [(0, 2), (2, 8), (8, 32), (32, 128), (128, 100000)]:
    m = (tsr >= lo) & (tsr < hi) & (tsr < 10**8)
    if m.sum():
        d = mfin[m]
        print(f"  {lo:3d}-{hi:5d} frames after reset: {m.mean()*100:5.1f}%  depth p50={np.nanpercentile(d,50)*1e3:7.2f}mm "
              f"p10={np.nanpercentile(d,10)*1e3:7.2f}mm  cube_w p90={np.percentile(cw[m],90):6.1f}  att mean={att[m].mean():.2f}")
