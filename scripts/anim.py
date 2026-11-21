"""
Procedural, physically-motivated animation for the zombie and the victim.

World layout: the street runs along Y.  The zombie stands at the origin
facing +X, notices the victim far down the street (-Y), turns and sprints
toward -Y at up to ~9.5 m/s (36 km/h), leaps, tackles the victim onto his
back and feeds on his neck (bite -> tear -> chew -> swallow -> repeat).

All motion is evaluated per frame and baked to keyframes.
"""
import math
import numpy as np
from mathutils import Vector, Quaternion, Matrix

from rigpose import RigPose, q_euler, q_axis, frame_rot

FPS = 24
NFRAMES = 720


# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def sstep(e0, e1, x):
    if e1 == e0:
        return 1.0 if x >= e1 else 0.0
    t = clamp((x - e0) / (e1 - e0))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def keys(t, pts):
    """piecewise smooth interpolation through [(t, value), ...]."""
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t <= t1:
            u = sstep(t0, t1, t)
            return v0 + (v1 - v0) * u
    return pts[-1][1]


def keys_lin(t, pts):
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return pts[-1][1]


class Noise:
    """smooth 1D noise, deterministic."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.tab = rng.uniform(-1, 1, 8192)

    def __call__(self, t, freq=1.0):
        x = t * freq * 8.0
        i = int(math.floor(x))
        f = x - i
        f = f * f * (3 - 2 * f)
        a = self.tab[i % 8192]
        b = self.tab[(i + 1) % 8192]
        return a + (b - a) * f

    def fbm(self, t, freq=1.0, oct=3):
        s, a, tot = 0.0, 1.0, 0.0
        for k in range(oct):
            s += a * self(t + 17.3 * k, freq * (2 ** k))
            tot += a
            a *= 0.5
        return s / tot


def twitch(t, times, dur=0.07, rel=0.5):
    """sharp onset, slower release spikes at given times -> value in [0,1]."""
    v = 0.0
    for t0 in times:
        if t0 <= t < t0 + dur + rel:
            if t < t0 + dur:
                v = max(v, sstep(t0, t0 + dur, t))
            else:
                v = max(v, 1 - sstep(t0 + dur, t0 + dur + rel, t))
    return v


def yawq(h_deg):
    return q_euler(0, 0, h_deg)


def fwd_of(h_deg):
    """world forward direction for heading h (character faces -Y at 0)."""
    r = math.radians(h_deg)
    return Vector((math.sin(r), -math.cos(r), 0))


def left_of(h_deg):
    r = math.radians(h_deg)
    return Vector((math.cos(r), math.sin(r), 0))


# ----------------------------------------------------------------------------
# Foot plants (no foot sliding: a planted foot is fixed in world space)
# ----------------------------------------------------------------------------
class Plant:
    def __init__(self, t_on, t_off, pos, yaw, kind):
        self.t_on, self.t_off = t_on, t_off
        self.pos = Vector((pos[0], pos[1], 0.0))
        self.yaw = yaw
        self.kind = kind


PITCH_PROFILES = {
    # (u, pitch deg) during stance; + = toes down / heel up
    "idle": [(0.0, 0.0), (1.0, 0.0)],
    "walk": [(0.0, -10.0), (0.15, 0.0), (0.65, 0.0), (1.0, 28.0)],
    "sprint": [(0.0, 10.0), (0.35, 3.0), (1.0, 50.0)],
    "push": [(0.0, 15.0), (0.5, 20.0), (1.0, 60.0)],
}
SWING = {
    "idle": dict(h=0.06, mid_pitch=8.0),
    "walk": dict(h=0.07, mid_pitch=8.0),
    "sprint": dict(h=0.52, mid_pitch=35.0),
    "push": dict(h=0.45, mid_pitch=40.0),
}


class FootTrack:
    def __init__(self, char_geo, side):
        self.g = char_geo
        self.side = side
        self.plants = []

    def add(self, t_on, t_off, pos, yaw, kind="idle"):
        self.plants.append(Plant(t_on, t_off, pos, yaw, kind))
        self.plants.sort(key=lambda p: p.t_on)

    def _stance(self, p, u):
        pitch = keys_lin(u, PITCH_PROFILES[p.kind])
        return self._place(p.pos, p.yaw, pitch)

    def _place(self, ground, yaw, pitch):
        g = self.g[self.side]
        Dq = yawq(yaw) @ q_euler(pitch, 0, 0)
        if pitch >= 0:
            piv_rest = g["ball"]
        else:
            piv_rest = g["heel"]
        Y = yawq(yaw)
        piv_world = Vector(ground) + Y @ Vector((piv_rest.x - g["ankle"].x, piv_rest.y - g["ankle"].y, 0)) \
            + Vector((0, 0, piv_rest.z))
        ankle = piv_world + Dq @ (g["ankle"] - piv_rest)
        return ankle, Dq, pitch

    def eval(self, t):
        P = self.plants
        if t <= P[0].t_on:
            return self._stance(P[0], 0.0) + (True,)
        for i, p in enumerate(P):
            if p.t_on <= t <= p.t_off:
                u = (t - p.t_on) / max(1e-6, p.t_off - p.t_on)
                return self._stance(p, u) + (True,)
            if i + 1 < len(P) and p.t_off < t < P[i + 1].t_on:
                q = P[i + 1]
                u = (t - p.t_off) / (q.t_on - p.t_off)
                a0, d0, p0 = self._stance(p, 1.0)
                a1, d1, p1 = self._stance(q, 0.0)
                sw = SWING[q.kind]
                if q.kind in ("sprint", "push"):
                    # heel recovery lags, then the knee drives forward
                    e = 0.5 * (u + sstep(0.0, 1.0, u))
                    hgt = sw["h"] * math.sin(math.pi * clamp(u) ** 0.7) ** 1.3
                else:
                    e = sstep(0.0, 1.0, u)
                    hgt = sw["h"] * math.sin(math.pi * u)
                pos = a0.lerp(a1, e)
                pos.z += hgt
                yaw = lerp(p.yaw, q.yaw, e)
                pitch = keys(u, [(0.0, p0), (0.45, sw["mid_pitch"]), (1.0, p1)])
                Dq = yawq(yaw) @ q_euler(pitch, 0, 0)
                return pos, Dq, pitch, False
        return self._stance(P[-1], 1.0) + (True,)


def foot_geometry(rp):
    g = {}
    for s in ("L", "R"):
        ankle = rp.h_rest["foot." + s].copy()
        ball = (rp.h_rest["toe1-1." + s] + rp.h_rest["toe5-1." + s]) * 0.5
        ball.z = 0.015
        heel = Vector((ankle.x, ankle.y + 0.055, 0.012))
        g[s] = dict(ankle=ankle, ball=ball, heel=heel)
    return g


# ----------------------------------------------------------------------------
# Generic body poser
# ----------------------------------------------------------------------------
SPINE = ["spine05", "spine04", "spine03", "spine02", "spine01"]
NECK = ["neck01", "neck02", "neck03"]
FINGERS = [f"finger{i}-{j}" for i in range(1, 6) for j in range(1, 4)]


class Poser:
    def __init__(self, char):
        self.c = char
        self.rp = RigPose(char.arm)
        self.rp.setup_poles()
        self.geo = foot_geometry(self.rp)
        self.feet = {"L": FootTrack(self.geo, "L"), "R": FootTrack(self.geo, "R")}
        hr = self.rp.h_rest
        self.hip_h = hr["root"].z
        self.root_rest = hr["root"].copy()
        self.leg_len = (hr["lowerleg01.L"] - hr["upperleg01.L"]).length + (hr["foot.L"] - hr["lowerleg01.L"]).length

    def local_rot(self, bone, q_local):
        return self.rp.B[bone] @ q_local @ self.rp.Binv[bone]

    def begin(self, root_pos, root_q):
        """root_pos: world position of pelvis (root head)."""
        pose = self.rp.new_pose()
        pose["root_loc"] = Vector(root_pos) - self.root_rest
        pose["R"]["root"] = root_q
        return pose

    def spine(self, pose, bend=0.0, side=0.0, twist=0.0, weights=(0.15, 0.2, 0.25, 0.2, 0.2)):
        for b, w in zip(SPINE, weights):
            pose["R"][b] = q_euler(bend * w, side * w, twist * w)

    def neck_head(self, pose, pitch=0.0, yaw=0.0, roll=0.0, neck_frac=0.45):
        for b in NECK:
            pose["R"][b] = q_euler(pitch * neck_frac / 3, roll * neck_frac / 3, yaw * neck_frac / 3)
        pose["R"]["head"] = q_euler(pitch * (1 - neck_frac), roll * (1 - neck_frac), yaw * (1 - neck_frac))

    def look_at(self, pose, target, amount=1.0, up=Vector((0, 0, 1)), roll=0.0, extra_pitch=0.0):
        """rotate neck+head so the face points at target (after fk)."""
        rp = self.rp
        rp.fk(pose)
        head_pos = pose["H"]["head"] + Vector((0, 0, 0.08))
        g = (Vector(target) - head_pos).normalized()
        Dp = pose["D"]["neck01"]
        D_parent = pose["D"][rp.parent["neck01"]]
        # current face direction with neutral neck
        rest_fwd = Vector((0, -1, 0))
        cur = D_parent @ rest_fwd
        cur_up = D_parent @ Vector((0, 0, 1))
        Dt = frame_rot(rest_fwd, Vector((0, 0, 1)), g, up)
        Dt = Dt @ q_euler(extra_pitch, roll, 0)
        Dfull = D_parent.slerp(Dt, amount) if amount < 1 else Dt
        parts = [0.18, 0.18, 0.18, 0.46]
        acc = 0.0
        for b, w in zip(NECK + ["head"], parts):
            acc += w
            rp.set_world(pose, b, D_parent.slerp(Dfull, acc))
        rp.fk(pose)

    def legs_ik(self, pose, t, knee_dir=None, override=None):
        rp = self.rp
        rp.fk(pose)
        out = {}
        for s in ("L", "R"):
            if override and s in override:
                ankle, Dq = override[s]
                contact = False
                pitch = 0.0
            else:
                ankle, Dq, pitch, contact = self.feet[s].eval(t)
            yaw_dir = Dq @ Vector((0, -1, 0))
            yaw_dir.z = 0
            kd = (knee_dir[s] if isinstance(knee_dir, dict) else knee_dir) if knee_dir is not None else yaw_dir
            kd = Vector(kd).normalized() + Vector((0, 0, 0.0))
            sgn = 1 if s == "L" else -1
            side_v = (Dq @ Vector((1, 0, 0)))
            pole = kd + side_v * sgn * 0.12
            rp.two_bone(pose, f"upperleg01.{s}", f"upperleg02.{s}", f"lowerleg01.{s}", f"lowerleg02.{s}",
                        f"foot.{s}", ankle, pole, end_rot=Dq)
            # keep toes on the ground when heel lifts
            tb = -max(0.0, pitch) * 0.85 if contact else -max(0.0, pitch) * 0.3
            for toe in (1, 2, 3, 4, 5):
                pose["R"][f"toe{toe}-1.{s}"] = q_euler(tb, 0, 0)
            out[s] = (ankle, contact)
        return out

    def arm_ik(self, pose, side, target, pole, hand_rot=None):
        rp = self.rp
        return rp.two_bone(pose, f"upperarm01.{side}", f"upperarm02.{side}", f"lowerarm01.{side}",
                           f"lowerarm02.{side}", f"wrist.{side}", target, pole, end_rot=hand_rot)

    def arm_pole(self, pose, side, back=0.8, out=0.7, down=0.4):
        f, u, l = self.chest_frame(pose)
        sgn = 1 if side == "L" else -1
        return -f * back + l * sgn * out - u * down

    def chest_frame(self, pose):
        D = pose["D"]["spine01"]
        return D @ Vector((0, -1, 0)), D @ Vector((0, 0, 1)), D @ Vector((1, 0, 0))

    def hand_from_chest(self, pose, side, fwd, up, lat):
        f, u, l = self.chest_frame(pose)
        sgn = 1 if side == "L" else -1
        S = pose["H"][f"upperarm01.{side}"]
        return S + f * fwd + u * up + l * (lat * sgn)

    def fingers(self, pose, side, curl=0.0, spread=0.0, thumb=0.0):
        sgn = 1 if side == "L" else -1
        for i in range(1, 6):
            for j in range(1, 4):
                b = f"finger{i}-{j}.{side}"
                if b not in self.rp.B:
                    continue
                if i == 1:
                    q = q_euler(thumb * 30 * (0.6 if j > 1 else 0.3), 0, 0)
                else:
                    c = curl * (55 if j == 1 else 70 if j == 2 else 50)
                    sp = spread * (i - 3) * 6 if j == 1 else 0
                    q = q_euler(c, 0, sp)
                pose["R"][b] = self.local_rot(b, q)

    def face(self, pose, jaw=0.0, snarl=0.0, lowlip=0.0, jaw_side=0.0, jaw_fwd=0.0, lids=0.0, brow=0.0,
             scream=0.0):
        R = pose["R"]
        R["jaw"] = q_euler(jaw, 0, jaw_side)
        R["special01"] = q_euler(-snarl * 2.5, 0, 0)
        R["special04"] = q_euler(lowlip * 12, 0, 0)
        for s, sg in (("L", 1), ("R", -1)):
            R["levator02." + s] = q_euler(-snarl * 10, 0, 0)
            R["levator03." + s] = q_euler(-snarl * 10, 0, 0)
            R["risorius03." + s] = q_euler(snarl * 5 + scream * 6, 0, sg * (snarl * 6 + scream * 9))
            R["orbicularis03." + s] = q_euler(lids * 22 - brow * 6, 0, 0)
            R["orbicularis04." + s] = q_euler(-lids * 10, 0, 0)
            R["oculi01." + s] = q_euler(brow * 10, 0, 0)

    def finish(self, frame, pose):
        self.rp.fk(pose)
        self.rp.store(frame, pose)


# ----------------------------------------------------------------------------
# The choreography
# ----------------------------------------------------------------------------
class Choreo:
    # key times (seconds)
    T_NOTICE = 5.62       # head snaps toward target
    T_TURN0 = 5.95
    T_TURN1 = 6.45
    T_LAUNCH = 6.80
    VMAX = 9.6
    T_LEAP = 13.30
    T_IMPACT = 13.62
    T_LAND = 14.25

    def __init__(self, zombie, victim):
        self.Z = Poser(zombie)
        self.V = Poser(victim)
        self.zc, self.vc = zombie, victim
        self.nz = Noise(11)
        self.nv = Noise(23)
        self.events = {}
        self._plan()

    # ---------------- planning ----------------
    def _plan(self):
        Z, V = self.Z, self.V
        zh = Z.hip_h
        # --- zombie sprint path (pelvis ground y), sampled finely
        dt = 1.0 / 240
        ts = np.arange(0, 31, dt)
        y = np.zeros_like(ts)
        v = np.zeros_like(ts)
        for i in range(1, len(ts)):
            t = ts[i]
            if t < self.T_LAUNCH:
                vel = 0.0
            else:
                u = (t - self.T_LAUNCH)
                vel = self.VMAX * (1 - math.exp(-u / 0.55))  # explosive acceleration
                if t > self.T_LEAP:
                    vel = self.VMAX * 0.95
                if t > self.T_IMPACT:
                    vel = max(0.0, self.VMAX * 0.95 * (1 - (t - self.T_IMPACT) / 0.7))
            v[i] = vel
            y[i] = y[i - 1] - vel * dt
        self.zt, self.zy, self.zv = ts, y, v
        y_imp = float(np.interp(self.T_IMPACT, ts, y))
        self.y_imp = y_imp
        # victim: pelvis 0.42 m ahead (-Y) of the zombie pelvis at impact
        self.vy_imp = y_imp - 0.42
        self.v_walk = 1.25
        self.T_V_NOTICE = 12.05
        self.T_V_STOP = 12.35
        back = 0.65
        self.vy_stop = self.vy_imp + back
        self.vy0 = self.vy_stop - self.v_walk * self.T_V_NOTICE - self.v_walk * 0.25
        # fall: victim slides back ~1.3 m while falling
        self.vy_ground = self.vy_imp - 1.15
        self._zombie_plants()
        self._victim_plants()

    def z_pelvis_y(self, t):
        return float(np.interp(t, self.zt, self.zy))

    def z_speed(self, t):
        return float(np.interp(t, self.zt, self.zv))

    def _zombie_plants(self):
        F = self.Z.feet
        # idle stance facing +X (heading 90): left foot at +Y side
        hd = 90.0
        L0 = Vector((0.05, 0.12, 0)); R0 = Vector((-0.06, -0.12, 0))
        F["L"].add(0.0, 2.35, L0, hd + 8, "idle")
        F["R"].add(0.0, 1.0, R0, hd - 12, "idle")
        # restless shuffle of the right foot
        R1 = Vector((0.02, -0.15, 0))
        F["R"].add(1.25, 5.95, R1, hd - 18, "idle")
        F["L"].add(2.6, 6.12, Vector((0.0, 0.13, 0)), hd + 5, "idle")
        # snap turn toward -Y (heading 0): quick steps
        F["R"].add(6.12, 6.75, Vector((-0.13, -0.05, 0)), -5, "idle")
        F["L"].add(6.32, 6.78, Vector((0.12, 0.18, 0)), 6, "push")   # back foot, pushes off
        # sprint plants from cadence model
        t = self.T_LAUNCH
        phase = 0.0
        dt = 1 / 480
        events = []
        # right foot is in front -> left foot swings first; start phase so L swings
        offs = {"L": 0.5, "R": 0.0}
        last = {"L": None, "R": None}
        cycle_start = {}
        prev_c = {s: phase + offs[s] for s in offs}
        state = {}
        tlist = np.arange(self.T_LAUNCH, self.T_LEAP + 0.001, dt)
        prev_phase = 0.25  # starts mid-way: R in stance, L in swing
        phase = prev_phase
        stance_on = {"L": None, "R": self.T_LAUNCH}
        for t in tlist:
            vel = self.z_speed(t)
            fr = 1.55 + 0.72 * clamp(vel / self.VMAX)
            beta = 0.52 - 0.28 * clamp(vel / self.VMAX)
            phase += fr * dt
            for s in ("L", "R"):
                c0 = (prev_phase + offs[s])
                c1 = (phase + offs[s])
                # touchdown when crossing integer
                if math.floor(c1) > math.floor(c0):
                    stance_on[s] = t
                    state[s] = beta
                if stance_on[s] is not None and (c1 % 1.0) >= state.get(s, beta) and (c0 % 1.0) < state.get(s, beta):
                    t_on = stance_on[s]
                    t_off = t
                    tm = 0.5 * (t_on + t_off)
                    y = self.z_pelvis_y(tm) - 0.06
                    x = 0.075 if s == "L" else -0.075
                    if t_on <= self.T_LAUNCH + 1e-3:
                        continue
                    kind = "sprint"
                    events.append((s, t_on, t_off, x, y))
                    stance_on[s] = None
            prev_phase = phase
        for s, t_on, t_off, x, y in events:
            F[s].add(t_on, t_off, Vector((x, y, 0)), 0.0 + (4 if s == "L" else -4), "sprint")
        # final push-off before the leap: last plant gets 'push'
        lastL = max([p for p in F["L"].plants], key=lambda p: p.t_on)
        lastR = max([p for p in F["R"].plants], key=lambda p: p.t_on)
        last = lastL if lastL.t_on > lastR.t_on else lastR
        last.kind = "push"
        self.z_last_plant = last
        self.z_last = {"L": lastL, "R": lastR}
        self.sprint_events = events

    def _victim_plants(self):
        F = self.V.feet
        hd = 180.0  # victim walks toward +Y (facing the zombie, eyes on phone)
        f = 0.9
        beta = 0.62
        T = self.T_V_NOTICE + 0.1
        phase = 0.0
        tcur = 0.0
        dt = 1 / 480
        stance_on = {"L": 0.0, "R": None}
        offs = {"L": 0.0, "R": 0.5}
        prev = 0.3
        ph = prev
        evs = []
        t = 0.0
        while t < T:
            ph += f * dt
            for s in ("L", "R"):
                c0, c1 = prev + offs[s], ph + offs[s]
                if math.floor(c1) > math.floor(c0):
                    stance_on[s] = t
                if stance_on[s] is not None and (c1 % 1) >= beta and (c0 % 1) < beta:
                    tm = 0.5 * (stance_on[s] + t)
                    yy = self.v_pelvis_y(tm) + 0.03
                    xx = -0.1 if s == "L" else 0.1   # facing +Y: left side is -X
                    evs.append((s, stance_on[s], t, xx, yy))
                    stance_on[s] = None
            prev = ph
            t += dt
        for s, a, b, xx, yy in evs:
            F[s].add(a, b, Vector((xx, yy, 0)), hd + (-6 if s == "L" else 6), "walk")
        # stop & freeze, then two steps backwards
        ys = self.vy_stop
        F["L"].add(self.T_V_STOP, 12.62, Vector((-0.11, ys + 0.05, 0)), hd - 6, "idle")
        F["R"].add(self.T_V_STOP + 0.05, 12.80, Vector((0.11, ys - 0.06, 0)), hd + 6, "idle")
        F["L"].add(12.85, 13.25, Vector((-0.12, ys - 0.35, 0)), hd - 10, "idle")
        F["R"].add(13.05, 13.60, Vector((0.12, ys - 0.62, 0)), hd + 10, "idle")
        F["L"].add(13.45, 13.66, Vector((-0.13, ys - 0.80, 0)), hd - 10, "idle")

    def v_pelvis_y(self, t):
        if t < self.T_V_NOTICE:
            return self.vy0 + self.v_walk * t
        if t < self.T_V_STOP + 0.2:
            u = (t - self.T_V_NOTICE) / (self.T_V_STOP + 0.2 - self.T_V_NOTICE)
            return self.vy0 + self.v_walk * self.T_V_NOTICE + self.v_walk * 0.25 * (1 - (1 - u) ** 2)
        y_st = self.vy0 + self.v_walk * self.T_V_NOTICE + self.v_walk * 0.25
        return keys(t, [(self.T_V_STOP + 0.2, y_st), (12.9, y_st), (13.62, self.vy_imp)])

    # ---------------- per-frame evaluation ----------------
    def run(self):
        for f in range(1, NFRAMES + 1):
            t = (f - 1) / FPS
            vpose = self.victim_body(t)
            zpose = self.zombie(t, vpose)
            self.victim_arms(t, vpose, zpose)
            self.V.finish(f, vpose)
            self.Z.finish(f, zpose)
            self.record(f, t, zpose, vpose)
        self.Z.rp.bake("zombie_anim")
        self.V.rp.bake("victim_anim")

    def record(self, f, t, zp, vp):
        ev = self.events.setdefault("frames", {})
        ev[f] = dict(
            z_head=tuple(zp["H"]["head"]), z_root=tuple(zp["H"]["root"]),
            v_head=tuple(vp["H"]["head"]), v_root=tuple(vp["H"]["root"]),
            z_mouth=tuple(self.Z.rp.point(zp, "jaw", self.zc.mouth)),
            v_bite=tuple(self.V.rp.point(vp, "neck01", self.vc.bite_point)),
            v_wristR=tuple(vp["H"]["wrist.R"]),
            v_wristR_D=tuple(vp["D"]["wrist.R"]),
            z_jaw=tuple(zp["H"]["jaw"]), z_jaw_D=tuple(zp["D"]["jaw"]),
            z_head_D=tuple(zp["D"]["head"]),
            z_chin=tuple(self.Z.rp.point(zp, "jaw", self.zc.mouth + Vector((0, 0.01, -0.045)))),
            info=dict(getattr(self, "_z_info", {})) if t >= self.T_IMPACT else {},
        )

    # ===================== ZOMBIE =====================
    def zombie(self, t, vpose):
        if t < self.T_LAUNCH:
            return self.z_idle_turn(t)
        if t < self.T_IMPACT:
            return self.z_sprint(t)
        return self.z_ground(t, vpose)

    def z_idle_turn(self, t):
        Z, n = self.Z, self.nz
        hd = keys(t, [(0, 90.0), (self.T_TURN0, 90.0), (self.T_TURN1, 0.0)])
        # pelvis sways between feet, slight stagger
        aL = Z.feet["L"].eval(t)[0]; aR = Z.feet["R"].eval(t)[0]
        mid = (aL + aR) * 0.5
        sway = 0.035 * math.sin(t * 2 * math.pi * 0.33) + 0.015 * n.fbm(t, 0.6)
        crouch = keys(t, [(0, 0.0), (6.3, 0.0), (6.55, 0.17), (self.T_LAUNCH, 0.14)])
        lat = left_of(hd)
        fw = fwd_of(hd)
        pos = Vector((mid.x, mid.y, 0)) + lat * sway + fw * (0.02 - crouch * 0.35)
        pos.z = Z.hip_h - 0.05 - crouch + 0.01 * math.sin(t * 2 * math.pi * 0.66)
        pitch = 10 + crouch * 140 + 2 * n(t, 0.5)
        roll = 4 * math.sin(t * 2 * math.pi * 0.33 + 0.6)
        rq = yawq(hd) @ q_euler(pitch, roll, 0)
        pose = Z.begin(pos, rq)
        # twitches (shoulder / head)
        tw = twitch(t, [1.32, 2.71, 3.38, 4.55, 5.02], 0.06, 0.45)
        breath = math.sin(t * 2 * math.pi * 0.28)
        notice = sstep(self.T_NOTICE, self.T_NOTICE + 0.09, t)  # very sharp head snap
        Z.spine(pose, bend=18 + crouch * 60 + breath * 2, side=-6 + 5 * tw, twist=-30 * notice * (1 - sstep(self.T_TURN0, self.T_TURN1, t)))
        Z.rp.fk(pose)
        # head: drooping, tilted, twitching; then locks onto victim
        vic = Vector((0, self.v_pelvis_y(t), 1.6))
        if notice > 0:
            Z.look_at(pose, vic, amount=notice, roll=-8 * (1 - crouch), extra_pitch=-6)
        else:
            Z.neck_head(pose, pitch=22 + 10 * tw + 3 * n(t, 0.7), yaw=8 * n.fbm(t, 0.25) - 25 * tw,
                        roll=-18 + 4 * n(t, 0.4) + 22 * tw)
        # arms hang limp, swing with sway; claws
        for s, sg in (("L", 1), ("R", -1)):
            wob = n(t + (3 if s == "L" else 7), 0.5)
            pull = crouch * 5.0   # arms swing back in the crouch
            tgt = Z.hand_from_chest(pose, s, 0.10 + 0.04 * wob - pull * 0.3, -0.55 + pull * 0.05, 0.05 + 0.02 * tw)
            Z.arm_ik(pose, s, tgt, pole=fwd_of(hd) * -1 + lat * sg * 0.6 + Vector((0, 0, -0.2)))
            Z.fingers(pose, s, curl=0.45 + 0.25 * tw + 0.1 * wob, spread=0.6, thumb=0.4)
        Z.legs_ik(pose, t)
        jaw = 5 + 3 * n(t, 0.4) + 6 * tw
        snarl = 0.15 * tw
        if t > self.T_NOTICE:
            k = sstep(self.T_NOTICE + 0.05, self.T_NOTICE + 0.35, t)
            jaw = lerp(jaw, 21 + 3 * math.sin(t * 9), k)
            snarl = lerp(snarl, 1.0, k)
        Z.face(pose, jaw=jaw, snarl=snarl, lowlip=0.7 * snarl, lids=-0.2 * sstep(5.6, 5.7, t), brow=-0.8 * notice)
        return pose

    def z_sprint(self, t):
        Z, n = self.Z, self.nz
        v = self.z_speed(t)
        k = clamp(v / self.VMAX)
        aL, _, _, cL = Z.feet["L"].eval(t)
        aR, _, _, cR = Z.feet["R"].eval(t)
        y = self.z_pelvis_y(t)
        # vertical oscillation: low in stance, high in flight
        in_stance = cL or cR
        x = 0.012 * math.sin(t * 2 * math.pi * 2.2) + 0.01 * n(t, 1.3)
        # pelvis height from support foot + flight bounce
        bob = 0.0
        sup = [a for a, c in ((aL, cL), (aR, cR)) if c]
        if sup:
            # pelvis over support: compress
            d = abs(sup[0].y - y)
            bob = -0.035 * (1 - clamp(d / 0.55)) * k
        else:
            bob = 0.035 * k
        base = Z.hip_h - lerp(0.14, 0.085, k)
        leap = 0.0
        if t > self.T_LEAP:
            u = (t - self.T_LEAP) / (self.T_IMPACT - self.T_LEAP)
            leap = 0.28 * math.sin(math.pi * clamp(u) * 0.8)
        pos = Vector((x, y, base + bob + leap))
        # strong forward lean (decreasing a bit at top speed), extra in the leap
        pitch = lerp(42, 26, k) + 4 * n(t, 1.1) + (sstep(self.T_LEAP, self.T_IMPACT, t) * 22)
        # pelvis yaw follows the legs
        phase_y = (aR.y - aL.y)
        yaw = -clamp(phase_y / 1.2, -1, 1) * 9 * k
        roll = 5 * math.sin(t * 2 * math.pi * 2.25) * k
        rq = yawq(yaw) @ q_euler(pitch, roll, 0)
        pose = Z.begin(pos, rq)
        Z.spine(pose, bend=12 + 6 * n(t, 0.9), side=-roll * 0.6, twist=-yaw * 2.4 + 6 * n(t, 1.7))
        Z.rp.fk(pose)
        # head juts forward and stays locked on the victim
        vic = Vector((0, self.v_pelvis_y(t), 1.45))
        Z.look_at(pose, vic, amount=1.0, roll=4 * n(t, 2.0), extra_pitch=-4)
        # arms: wild, high, clawing swing opposite to legs
        f_ = Z.chest_frame(pose)[0]
        leap_k = sstep(self.T_LEAP - 0.15, self.T_LEAP + 0.08, t)
        for s, sg in (("L", 1), ("R", -1)):
            # arm forward when the opposite foot is forward
            aOpp = aR if s == "L" else aL
            aSame = aL if s == "L" else aR
            a = clamp((aSame.y - aOpp.y) / 1.1, -1, 1)  # >0 -> same-side foot is behind -> arm forward
            amp = 1.25
            fw = 0.05 + 0.33 * a * amp
            fw = 0.05 + (0.33 * max(a, 0) + 0.24 * min(a, 0)) * amp
            up = -0.30 + 0.24 * max(a, 0) + 0.02 * max(-a, 0)
            lat = 0.10 - 0.12 * max(a, 0) + 0.06 * max(-a, 0)
            fw += 0.05 * n(t + sg * 5, 2.5); up += 0.05 * n(t + sg * 9, 2.2)
            # reach for the prey in the leap
            fw = lerp(fw, 0.62, leap_k); up = lerp(up, 0.05, leap_k); lat = lerp(lat, 0.12, leap_k)
            tgt = Z.hand_from_chest(pose, s, fw, up, lat)
            pole = Vector((0, 1, -0.4)) + Vector((sg * 0.8, 0, 0))
            Z.arm_ik(pose, s, tgt, pole)
            Z.fingers(pose, s, curl=lerp(0.6, 0.35, leap_k) + 0.15 * n(t + sg, 3), spread=0.8, thumb=0.5)
        ov = {}
        for s_, sg in (("L", 1), ("R", -1)):
            lp = self.z_last[s_]
            if t > lp.t_off:
                u = sstep(lp.t_off, lp.t_off + 0.22, t)
                hip = pose["H"]["upperleg01." + s_]
                a_end = hip + Vector((sg * 0.12, 0.72 + 0.1 * sg, -0.36 + 0.14 * sg))
                a0, D0, p0, _ = Z.feet[s_].eval(lp.t_off)
                ov[s_] = (a0.lerp(a_end, u), q_euler(lerp(p0, 65, u), 0, 0))
        Z.legs_ik(pose, t, override=ov or None)
        Z.face(pose, jaw=20 + 6 * math.sin(t * 11) + 4 * n(t, 3), snarl=1.0, lowlip=0.8, lids=-0.15, brow=-1.0)
        return pose

    # ---- feeding ----
    def bite_state(self, t):
        """returns dict: b (0 head raised .. 1 mouth on wound), jaw, shake, chew."""
        b = keys(t, [(14.25, 0.0), (14.9, 0.15), (15.05, 0.25), (15.22, 1.0), (16.6, 1.0), (17.25, 0.75),
                     (17.45, 0.0), (21.3, 0.0), (21.55, 0.25), (21.95, 1.0), (24.6, 1.0), (25.0, 0.7),
                     (25.2, 0.0), (30.5, 0.0)])
        return b

    def z_ground(self, t, vpose):
        Z, V, n = self.Z, self.V, self.nz
        rp = Z.rp
        # victim reference points
        bite = V.rp.point(vpose, "neck01", self.vc.bite_point)
        v_root = vpose["H"]["root"]
        b = self.bite_state(t)
        # tear shakes (head yaw oscillation)
        shake = 0.0
        for (t0, t1, amp, fq) in ((15.25, 16.6, 8, 6.0), (16.6, 17.4, 22, 4.2), (22.0, 23.0, 9, 6.5),
                                  (23.6, 24.0, 10, 6.0), (24.0, 25.1, 25, 4.0)):
            if t0 <= t <= t1:
                env = sstep(t0, t0 + 0.12, t) * (1 - sstep(t1 - 0.1, t1, t))
                shake += amp * env * math.sin((t - t0) * 2 * math.pi * fq)
        # chewing cycles (when head raised)
        chew_on = (sstep(17.5, 17.8, t) * (1 - sstep(21.1, 21.3, t))) + sstep(25.3, 25.6, t)
        chew_f = 1.55
        cph = (t * chew_f) % 1.0
        # opening slow (60%), closing fast & powerful (40%)
        if cph < 0.6:
            chew = sstep(0, 0.6, cph)
        else:
            chew = 1 - sstep(0.6, 1.0, cph) ** 0.7
        chew_amp = 11 + 4 * n(t, 0.3)
        swallow = twitch(t, [21.05, 28.6], 0.12, 0.35)
        # impact/landing (13.62-14.25): falling with the victim
        land = sstep(self.T_IMPACT, self.T_LAND, t)
        # body posture: kneeling straddle, torso pitched over the victim
        torso_pitch = lerp(lerp(60, 82, land), lerp(58, 104, b), sstep(self.T_LAND, 14.6, t))
        torso_pitch += -8 * swallow + 3 * n(t, 0.8)
        # pull-back during tearing
        pull = 0.0
        for (t0, t1) in ((16.6, 17.45), (24.0, 25.2)):
            if t0 <= t <= t1:
                pull = max(pull, sstep(t0, t1, t))
        torso_pitch -= 14 * pull
        yaw = 6 * n(t, 0.35) + 0.3 * shake
        rq = yawq(yaw) @ q_euler(torso_pitch, 4 * n(t, 0.5) + 0.25 * shake, 0)
        pose = Z.begin(Vector((0, 0, 0.6)), rq)
        Z.spine(pose, bend=lerp(12, 34, b) - 10 * pull, side=0.4 * shake, twist=0.6 * shake)
        # neck/head: dive toward the wound when biting, raised & chewing otherwise
        head_pitch = lerp(-38, 18, b) + chew_on * (4 * chew) - 12 * swallow - 10 * pull
        Z.neck_head(pose, pitch=head_pitch, yaw=shake + 14 * (1 - b) * n(t, 0.25), roll=lerp(-10, -38, b) + 0.5 * shake,
                    neck_frac=0.55)
        rp.fk(pose)
        mouth = rp.point(pose, "jaw", self.zc.mouth)
        # desired mouth position
        if t < self.T_LAND:
            # falling: zombie pelvis follows the impact trajectory
            u = sstep(self.T_IMPACT, self.T_LAND, t)
            yz = lerp(self.y_imp, self.vy_ground + 0.62, u)
            zz = lerp(Z.hip_h - 0.0, 0.62, u) + 0.25 * math.sin(math.pi * u) * (1 - u)
            target_root = Vector((0.03, yz, zz))
            delta = target_root - pose["H"]["root"]
        else:
            hover = bite + Vector((0.04, 0.12, 0.40 + 0.03 * chew_on)) + Vector((0, 0, 0.03 * n(t, 0.6)))
            goal = hover.lerp(bite + Vector((0.01, 0.0, 0.025)), b)
            goal += Vector((0, 0.06 * pull, 0.03 * pull))
            delta = goal - mouth
            # blend in from the landing pose
            k = sstep(self.T_LAND, 14.7, t)
            root_land = Vector((0.03, self.vy_ground + 0.62, 0.62)) - pose["H"]["root"]
            delta = root_land.lerp(delta, k)
        pose["root_loc"] += delta
        rp.fk(pose)
        # legs: knees on the ground straddling the victim, shins flat behind
        hipL = pose["H"]["upperleg01.L"]
        ov_pts = {}
        L1 = (rp.h_rest["lowerleg01.L"] - rp.h_rest["upperleg01.L"]).length
        for s, sg in (("L", 1), ("R", -1)):
            hip = pose["H"]["upperleg01." + s]
            hz = max(0.06, hip.z - 0.06)
            horiz = math.sqrt(max(0.0, L1 * L1 - hz * hz))
            knee = Vector((hip.x + sg * 0.20, hip.y + horiz * 0.85, 0.06))
            # lift a knee occasionally (shifting weight)
            knee.z += 0.05 * max(0, n(t + sg * 4, 0.4))
            ankle = knee + Vector((sg * 0.05, 0.42, 0.02))
            if t < self.T_LAND:
                u = sstep(self.T_IMPACT, self.T_LAND, t)
                a_air = hip + Vector((sg * 0.15, 0.7, -0.25))
                knee = (hip + Vector((sg * 0.18, 0.3, -0.35))).lerp(knee, u)
                ankle = a_air.lerp(ankle, u)
            Dfoot = q_euler(lerp(70, 110, sstep(self.T_IMPACT, self.T_LAND, t)), 0, 0)
            rp.limb_points(pose, f"upperleg01.{s}", f"upperleg02.{s}", f"lowerleg01.{s}", f"lowerleg02.{s}",
                           f"foot.{s}", knee, ankle, Vector((0, -1, 0.2)), end_rot=Dfoot)
        rp.fk(pose)
        # hands: pin the victim's shoulders; right hand grabs hair/head during tearing
        vsL = V.rp.point(vpose, "clavicle.L", V.rp.t_rest["clavicle.L"]) + Vector((0, 0, 0.07))
        vsR = V.rp.point(vpose, "clavicle.R", V.rp.t_rest["clavicle.R"]) + Vector((0, 0, 0.07))
        vhead = vpose["H"]["head"] + Vector((0, 0, 0.09))
        grab_head = sstep(16.4, 16.7, t) * (1 - sstep(17.4, 17.8, t)) + sstep(23.8, 24.1, t) * (1 - sstep(25.2, 25.6, t))
        settle = sstep(self.T_IMPACT + 0.2, 14.6, t)
        for s, sg in (("L", 1), ("R", -1)):
            # face to face: zombie's left hand -> victim's right shoulder and vice versa
            if s == "L":
                tgt = vsR
            else:
                tgt = vsL.lerp(vhead + Vector((-0.04, -0.05, 0.0)), grab_head)
            air = Z.hand_from_chest(pose, s, 0.55, -0.05, 0.15)
            tgt = air.lerp(tgt + Vector((0.02 * n(t + sg, 1.5), 0.0, 0.02 * n(t + 3 * sg, 1.7))), settle)
            Z.arm_ik(pose, s, tgt, Vector((sg * 1.0, 0.4, 0.6)))
            Z.fingers(pose, s, curl=0.75 + 0.15 * n(t + sg * 2, 2), spread=0.7, thumb=0.6)
        # face
        if b > 0.5:
            # mouth on the wound: wide during the lunge, clamped while tearing
            lunge_open = 1 - sstep(15.15, 15.27, t) if t < 18 else 1 - sstep(21.85, 21.98, t)
            jaw = lerp(5 + 2 * abs(math.sin(t * 13)), 26, lunge_open)
            snarl = 1.0
        else:
            jaw_raise = 20 * (1 - chew_on) * sstep(14.5, 15.0, t)
            jaw = chew_on * (3 + chew_amp * chew) + jaw_raise + (1 - chew_on) * 6
            snarl = 0.55 + 0.45 * (1 - chew_on) + 0.2 * chew
        side_grind = chew_on * 3.5 * math.sin(t * chew_f * 2 * math.pi + 1.2) * (1 if int(t * chew_f) % 2 else -1)
        Z.face(pose, jaw=jaw, snarl=snarl, lowlip=0.6 + 0.3 * chew_on * chew, jaw_side=side_grind,
               lids=0.25 * chew_on, brow=-0.8)
        # store some info for FX
        self._z_info = dict(b=b, chew=chew * chew_on, shake=shake, pull=pull)
        return pose

    # ===================== VICTIM =====================
    def victim_body(self, t):
        V, n = self.V, self.nv
        if t < self.T_IMPACT:
            return self.v_standing(t)
        return self.v_fall_ground(t)

    def v_standing(self, t):
        V, n = self.V, self.nv
        aL, _, _, cL = V.feet["L"].eval(t)
        aR, _, _, cR = V.feet["R"].eval(t)
        y = self.v_pelvis_y(t)
        mid = (aL + aR) * 0.5
        walking = 1 - sstep(self.T_V_NOTICE, self.T_V_STOP, t)
        # pelvis bob: lowest at double support
        dy = abs(aL.y - aR.y)
        bob = 0.025 * clamp(1 - dy / 0.7) * walking
        x = lerp(mid.x, mid.x, 1) + 0.025 * math.sin(t * 2 * math.pi * 0.9) * walking
        flinch = sstep(self.T_V_NOTICE, self.T_V_NOTICE + 0.25, t)
        back = sstep(12.85, 13.6, t)
        pos = Vector((x * 0.6, y, V.hip_h - 0.03 - bob - 0.04 * flinch))
        pitch = 4 * walking - 6 * flinch - 6 * back
        yaw = 180 + 5 * math.sin(t * 2 * math.pi * 0.9) * walking
        rq = yawq(yaw) @ q_euler(pitch, 2 * math.sin(t * 2 * math.pi * 0.9) * walking, 0)
        pose = V.begin(pos, rq)
        V.spine(pose, bend=6 * walking - 10 * flinch, twist=-5 * math.sin(t * 2 * math.pi * 0.9) * walking)
        V.rp.fk(pose)
        zpos = Vector((0, self.z_pelvis_y(t), 1.5))
        if t < self.T_V_NOTICE:
            V.neck_head(pose, pitch=28 + 2 * n(t, 0.5), yaw=3 * n(t, 0.3), roll=2)
        else:
            look = sstep(self.T_V_NOTICE, self.T_V_NOTICE + 0.18, t)
            V.neck_head(pose, pitch=28 * (1 - look), yaw=3 * (1 - look), roll=2)
            if look > 0:
                V.look_at(pose, zpos, amount=look, extra_pitch=-5 * look)
        self._v_arm_state = (walking, flinch, back)
        V.legs_ik(pose, t)
        scream = sstep(12.9, 13.2, t)
        V.face(pose, jaw=4 + 18 * flinch + 6 * scream, lowlip=0.3 * flinch, brow=1.0 * flinch, lids=-0.4 * flinch,
               scream=scream)
        return pose

    def v_fall_ground(self, t):
        V, n = self.V, self.nv
        u = sstep(self.T_IMPACT, self.T_LAND, t)
        uf = clamp((t - self.T_IMPACT) / (self.T_LAND - self.T_IMPACT))
        # pivot backward: pitch from -6 to -90 (face up), body slides back
        fall_pitch = -6 - 84 * (uf ** 1.6)
        bounce = 0.0
        if t > self.T_LAND:
            bounce = 0.035 * math.exp(-(t - self.T_LAND) * 9) * abs(math.sin((t - self.T_LAND) * 25))
        y = lerp(self.vy_imp, self.vy_ground, 1 - (1 - uf) ** 2)
        z_stand = V.hip_h - 0.08
        z_ground = 0.13
        z = lerp(z_stand, z_ground, uf ** 1.4) + bounce
        # struggle: pelvis bucks, twists
        str_k = self.struggle(t)
        twist = 0.0
        if t > self.T_LAND:
            twist = 10 * n(t, 0.9) * str_k
            z += 0.03 * max(0, n(t + 5, 1.5)) * str_k
        rq = yawq(180 + twist * 0.5) @ q_euler(fall_pitch, twist * 0.4, 0)
        pose = V.begin(Vector((0.0, y, z)), rq)
        # curl up while falling then arch/struggle
        V.spine(pose, bend=lerp(-10, 8, u) + 8 * str_k * n(t, 1.1), side=6 * str_k * n(t + 2, 0.8), twist=twist)
        # head: whiplash, then turned away (to his right) exposing the left neck
        whip = math.exp(-max(0, t - self.T_LAND) * 6) * math.sin(max(0, t - self.T_LAND) * 30) * 12
        turn = sstep(self.T_LAND, 14.9, t)
        V.neck_head(pose, pitch=lerp(25, -18, u) + whip + 4 * str_k * n(t, 2.2), yaw=-40 * turn + 10 * str_k * n(t, 1.3),
                    roll=-15 * turn)
        V.rp.fk(pose)
        # legs: thrown up during the fall, then kicking / bending
        ov = {}
        for s, sg in (("L", -1), ("R", 1)):   # facing +Y: left is -X
            hip = pose["H"]["upperleg01." + s]
            if t < self.T_LAND + 0.05:
                a_stand = V.feet[s].eval(self.T_IMPACT)[0]
                a_air = hip + Vector((sg * 0.12, 0.65, 0.15))
                a = a_stand.lerp(a_air, sstep(0, 1, uf))
                ov[s] = (a, yawq(180) @ q_euler(lerp(0, -30, uf), 0, 0))
            else:
                kick_ph = t * 1.7 + (0 if s == "L" else 0.5)
                kick = (math.sin(kick_ph * 2 * math.pi) * 0.5 + 0.5) * str_k
                bend = 0.25 + 0.35 * kick + 0.15 * (1 - str_k)
                knee_up = bend
                # foot position: along the body axis toward +Y, sliding on ground
                a = hip + Vector((sg * (0.18 + 0.06 * n(t + sg, 0.5)), 0.85 - 0.45 * knee_up, -0.12 + 0.05 * kick))
                a.z = max(0.07, a.z)
                ov[s] = (a, yawq(180) @ q_euler(-40 + 30 * kick, 0, 0))
        V.legs_ik(pose, t, knee_dir={"L": Vector((-0.3, 0, 1)), "R": Vector((0.3, 0, 1))}, override=ov)
        scream = (sstep(14.3, 14.6, t) * (1 - sstep(21.5, 24.0, t))) * (0.7 + 0.3 * abs(n(t, 3)))
        gasp = sstep(24.0, 25.0, t) * (1 - sstep(27.0, 28.5, t))
        V.face(pose, jaw=6 + 20 * scream + 8 * gasp, lowlip=0.5 * scream, brow=1.0 * scream,
               lids=-0.5 * scream + 0.6 * sstep(26.5, 28.5, t), scream=scream)
        return pose

    def struggle(self, t):
        return keys(t, [(14.25, 0.4), (14.6, 1.0), (19.5, 1.0), (22.5, 0.65), (25.0, 0.3), (27.0, 0.1), (28.5, 0.0)])

    def victim_arms(self, t, vp, zp):
        V, n = self.V, self.nv
        V.rp.fk(vp)
        if t < self.T_IMPACT:
            walking, flinch, back = self._v_arm_state
            for s, sg in (("L", -1), ("R", 1)):
                if s == "R":
                    # phone held in front of the chest
                    phone = V.hand_from_chest(vp, s, 0.27, -0.22, -0.10)
                    defend = V.hand_from_chest(vp, s, 0.32, 0.02, 0.12)
                    tgt = phone.lerp(defend, sstep(12.8, 13.35, t))
                    tgt = tgt.lerp(V.hand_from_chest(vp, s, 0.12, -0.2, 0.2), flinch * (1 - sstep(12.8, 13.35, t)) * 0.4)
                else:
                    ph = math.sin(t * 2 * math.pi * 0.9) * walking
                    swing = V.hand_from_chest(vp, s, 0.06 + 0.13 * ph, -0.52, 0.05)
                    defend = V.hand_from_chest(vp, s, 0.30, 0.05, 0.15)
                    tgt = swing.lerp(defend, sstep(12.6, 13.2, t))
                pole = V.arm_pole(vp, s)
                if s == "R":
                    # wrist turned so the phone screen faces his eyes
                    V.arm_ik(vp, s, tgt, pole)
                    D = vp["D"]["lowerarm02.R"]
                    V.rp.set_world(vp, "wrist.R", D @ q_axis(V.rp.B["wrist.R"] @ Vector((0, 1, 0)), 0) )
                    V.fingers(vp, s, curl=0.55 * (1 - sstep(12.8, 13.2, t)) + 0.2, spread=0.2, thumb=0.6)
                else:
                    V.arm_ik(vp, s, tgt, pole)
                    V.fingers(vp, s, curl=0.25 + 0.4 * sstep(12.6, 13.2, t), spread=0.5 * flinch)
            return
        # on the ground: flailing then pushing the zombie's head / shoulders
        u = sstep(self.T_IMPACT, self.T_LAND, t)
        str_k = self.struggle(t)
        z_head = zp["H"]["head"] + zp["D"]["head"] @ Vector((0, -0.04, 0.05))
        z_shL = zp["H"]["upperarm01.L"]
        z_shR = zp["H"]["upperarm01.R"]
        for s, sg in (("L", -1), ("R", 1)):
            fl = V.hand_from_chest(vp, s, 0.25 + 0.2 * math.sin(t * 9 + sg), 0.45, 0.35)
            if s == "R":
                push = z_head + Vector((0.03 * n(t, 3), 0.0, 0.0))
            else:
                push = z_shR + Vector((0.0, 0.02, 0.05))
            # pushing hands jitter (effort), arms collapse as he weakens
            push += Vector((0.03 * n(t + sg, 4) * str_k, 0.03 * n(t + 2 * sg, 3.5) * str_k, 0.02 * n(t + 5, 4) * str_k))
            limp = V.hand_from_chest(vp, s, 0.05, -0.15 if s == "R" else -0.4, 0.55)
            limp.z = max(0.05, limp.z)
            tgt = fl.lerp(push, sstep(self.T_LAND - 0.1, 14.8, t))
            tgt = limp.lerp(tgt, clamp(str_k * 1.4))
            V.arm_ik(vp, s, tgt, Vector((sg * 1.0, 0.0, -0.6)))
            V.fingers(vp, s, curl=0.35 + 0.35 * str_k, spread=0.6 * str_k, thumb=0.4)
