"""Attribution + geometry forensics: per-world attempts, offender qpos reproduction, hull measurements."""
import glob
import os
import numpy as np
import mujoco

BASE = os.path.dirname(os.path.abspath(__file__))
RICH = os.path.join(BASE, "rich")

parts = sorted(glob.glob(os.path.join(RICH, "rich_*.npz")))
data = [np.load(p, allow_pickle=True) for p in parts]
dt = np.concatenate([d["dt"] for d in data])
att = np.concatenate([d["att"] for d in data])
ncon = np.concatenate([d["ncon"] for d in data])
mind = np.concatenate([d["mindist"] for d in data])
qid = sum([list(d["qid"]) for d in data], [])
qpos = sum([list(d["qpos"]) for d in data], [])
F, N = att.shape
names = open(os.path.join(RICH, "geom_names.txt")).read().splitlines()

print("=" * 70)
print("PART 1: PER-WORLD WORK ATTRIBUTION")
print("=" * 70)
fmax = att.max(1)
print(f"attempts/world/frame: mean={att.mean():.2f} p50={np.percentile(att,50):.0f} p99={np.percentile(att,99):.0f}")
print(f"frame max attempts:   mean={fmax.mean():.1f} p50={np.percentile(fmax,50):.0f} p99={np.percentile(fmax,99):.0f} max={fmax.max()}")
print(f"frame p99.9-world:    mean={np.percentile(att,99.9,axis=1).mean():.1f}")
share = att.max(1) / np.maximum(att.sum(1), 1)
kmax = att.argmax(1)
print(f"worlds at frame-max needing >2x median-world attempts: "
      f"{np.mean(fmax > 2*np.maximum(np.percentile(att,50,axis=1),1))*100:.0f}% of frames")
# what does the binding (max-attempt) world look like?
bmask = np.zeros_like(att, dtype=bool)
bmask[np.arange(F), kmax] = True
print(f"\nbinding world state: in-contact {np.mean(ncon[bmask]>0)*100:.0f}% "
      f"(vs all-world {np.mean(ncon>0)*100:.0f}%)")
bm = mind[bmask]
am = mind[np.isfinite(mind)]
bm = bm[np.isfinite(bm)]
print(f"binding world deepest-pen p50={np.percentile(bm,50)*1e3:.1f}mm (all worlds {np.percentile(am,50)*1e3:.1f}mm)")
print(f"binding world ncon p50={np.percentile(ncon[bmask],50):.0f} (all {np.percentile(ncon,50):.0f})")
# distinct binding worlds
uniq = len(set(kmax.tolist()))
print(f"distinct binding worlds across {F} frames: {uniq}")
# how much of total batch work is the tail?
tot = att.sum()
print(f"work share: top-1 world/frame {att.max(1).sum()/tot*100:.1f}%, top-8/frame "
      f"{np.sort(att,1)[:,-8:].sum()/tot*100:.1f}% of all attempts")

print()
print("=" * 70)
print("PART 2: HULL GEOMETRY MEASUREMENTS (model.mjb = what the solver simulates)")
print("=" * 70)
mjm = mujoco.MjModel.from_binary_path(os.path.join(RICH, "model.mjb"))
print(f"ngeom={mjm.ngeom} nmesh={mjm.nmesh} nq={mjm.nq} nv={mjm.nv}")


def geom_id_by_name(frag):
    ids = []
    for g in range(mjm.ngeom):
        nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g) or names[g] if g < len(names) else ""
        if nm and frag in nm:
            ids.append((g, nm))
    return ids


def hull_extents(g):
    t = mjm.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = mjm.geom_dataid[g]
        va, vn = mjm.mesh_vertadr[mid], mjm.mesh_vertnum[mid]
        v = mjm.mesh_vert[va:va + vn]
        return v.max(0) - v.min(0), vn
    return mjm.geom_size[g] * 2, -1


for frag in ["object", "middle_link_1", "index_link_1", "middle_link_2", "palm", "middle_link_3"]:
    for g, nm in geom_id_by_name(frag)[:2]:
        ext, vn = hull_extents(g)
        print(f"  {nm}: type={mujoco.mjtGeom(mjm.geom_type[g]).name} extents={np.array2string(ext*1e3, precision=1)}mm verts={vn} "
              f"margin={mjm.geom_margin[g]:.4f} solimp_width={mjm.geom_solimp[g][2]:.4f} solref={mjm.geom_solref[g]}")

print()
print("=" * 70)
print("PART 3: CPU-MUJOCO REPRODUCTION OF BINDING-WORLD CONFIGURATIONS")
print("=" * 70)
mjd = mujoco.MjData(mjm)
# take the 12 frames with the highest frame-max attempts; reproduce their top world
hard = np.argsort(fmax)[::-1][:12]
for f in hard:
    w = kmax[f]
    ids = list(qid[f])
    if w not in ids:
        continue
    q = qpos[f][ids.index(w)]
    mjd.qpos[:] = q
    mjd.qvel[:] = 0
    mujoco.mj_forward(mjm, mjd)
    cons = []
    for c in range(mjd.ncon):
        con = mjd.contact[c]
        g1n = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"g{con.geom1}"
        g2n = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"g{con.geom2}"
        cons.append((con.dist, g1n.split("/")[-2] if "/" in g1n else g1n,
                     g2n.split("/")[-2] if "/" in g2n else g2n, con.frame[:3].copy()))
    cons.sort()
    print(f"frame {f} world {w} attempts={att[f,w]} dt_end={dt[f,w]:.2e} ncon(cpu)={mjd.ncon}")
    for dist, a, b, nrm in cons[:6]:
        print(f"    dist={dist*1e3:7.2f}mm  {a} <-> {b}  normal={np.array2string(nrm, precision=2)}")
    # conditioning proxy: max pairwise normal opposition among deep contacts
    deepn = [nrm for dist, a, b, nrm in cons if dist < -1e-3]
    worst = 0.0
    for i in range(len(deepn)):
        for j in range(i + 1, len(deepn)):
            worst = max(worst, -float(np.dot(deepn[i], deepn[j])))
    if deepn:
        print(f"    max normal opposition among deep contacts: {worst:.2f} (1.0 = directly opposing)")
