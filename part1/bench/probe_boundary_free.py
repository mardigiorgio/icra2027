"""A/B at N worlds: 20 x 0.1 s boundaries vs one boundary-free 2 s march,
icf-adaptive. Manual capture: warm-up is a 10 ms eager march (module
loads only), each variant is then captured once (capture cost is
independent of boundary length: the march body records once inside the
conditional while) and the timed pass replays graphs only."""
import sys, time
import warp as wp
sys.path.insert(0, "/home/mdigiorgio/Documents/code/icra2027")
from part1.bench.four_arms import _icf, _icf_params, K_INIT, NEWTON_KAPPA, NEWTON_TOL_FLOOR, DT_INNER_MIN, build_model
from part1.scenes.cenic_scenes import SCENES
import newton

N = int(sys.argv[1])
tol = 1e-3
model = build_model(N, scene="hard-clutter")
icf = _icf()
solver = icf.SolverICFAdaptive(
    model,
    params=_icf_params({**SCENES["hard-clutter"].icf, "newton_tolerance": max(NEWTON_KAPPA * tol, NEWTON_TOL_FLOOR)}),
    adaptive=icf.IcfAdaptiveParams(tol=tol, dt_inner_init=K_INIT * 0.1,
                                   dt_inner_min=DT_INNER_MIN, dt_inner_max=0.1, max_substeps=4096),
)
pipeline = newton.CollisionPipeline(model)
contacts = pipeline.contacts()
solver.attach_collision_pipeline(pipeline)

init = model.state()
s0, s1, ctrl = model.state(), model.state(), model.control()

def reset():
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        a, b = getattr(s0, name, None), getattr(init, name, None)
        if a is not None and b is not None and a.size:
            wp.copy(a, b)

def body(dt, a, b):
    pipeline.collide(a, contacts)
    solver.step(a, b, ctrl, contacts, dt)

# warm: tiny eager march, loads every module
body(0.01, s0, s1)
reset(); wp.synchronize()

def capture(dt, a, b):
    with wp.ScopedCapture() as cap:
        body(dt, a, b)
    return cap.graph

gA0 = capture(0.1, s0, s1)
gA1 = capture(0.1, s1, s0)
gB0 = capture(2.0, s0, s1)
reset(); wp.synchronize()
t0 = time.perf_counter()
for k in range(20):
    wp.capture_launch(gA0 if k % 2 == 0 else gA1)
wp.synchronize()
a_wall = time.perf_counter() - t0
reset(); wp.synchronize()
t0 = time.perf_counter()
wp.capture_launch(gB0)
wp.synchronize()
b_wall = time.perf_counter() - t0
print(f"N={N}: 20 x 0.1 s boundaries = {a_wall:.2f}s | one boundary-free 2 s march = {b_wall:.2f}s | speedup {a_wall/b_wall:.2f}x")
