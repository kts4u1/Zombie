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


def spline(t, pts, m0=None, m1=None):
    """C1 Catmull-Rom (non-uniform knots) through [(t, value), ...]; value float or Vector.
    m0 / m1: optional end tangents (d value / d t).  No stop-and-go at the knots."""
    n = len(pts)
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    i = 0
    while pts[i + 1][0] < t:
        i += 1
    (t0, v0), (t1, v1) = pts[i], pts[i + 1]

    def tan(j):
        if j == 0:
            return m0 if m0 is not None else (pts[1][1] - pts[0][1]) * (1.0 / (pts[1][0] - pts[0][0]))
        if j == n - 1:
            return m1 if m1 is not None else (pts[-1][1] - pts[-2][1]) * (1.0 / (pts[-1][0] - pts[-2][0]))
        return (pts[j + 1][1] - pts[j - 1][1]) * (1.0 / (pts[j + 1][0] - pts[j - 1][0]))

    h = t1 - t0
    s = (t - t0) / h
    s2, s3 = s * s, s * s * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return v0 * h00 + tan(i) * (h10 * h) + v1 * h01 + tan(i + 1) * (h11 * h)


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
        self.sk_frames = {}

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
        if up is None:
            # reference 'up' from the chest: well defined even when looking straight down
            up = cur_up
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

    def expr(self, pose, units):
        """facial expression units (shape keys, 0..1.5) for this frame."""
        sk = pose.setdefault("SK", {})
        for k, v in units.items():
            if v > 1e-4:
                sk[k] = min(1.5, sk.get(k, 0.0) + v)

    def eyes_at(self, pose, target, amount=1.0, limit=30.0):
        """aim both eyeballs at a world point (clamped to a natural range)."""
        rp = self.rp
        rp.fk(pose)
        Dh = pose["D"]["head"]
        for s in ("L", "R"):
            b = "eye." + s
            if b not in rp.B:
                continue
            d = Dh.inverted() @ (Vector(target) - pose["H"][b]).normalized()
            q = Vector((0, -1, 0)).rotation_difference(d)
            ang = q.angle
            if ang > math.radians(limit):
                q = Quaternion(q.axis, math.radians(limit))
            q = Quaternion().slerp(q, amount)
            rp.set_world(pose, b, Dh @ q)

    def eyes_dir(self, pose, pitch=0.0, yaw=0.0):
        """eyes rotated relative to the head (deg; + pitch looks down)."""
        Dh = pose["D"]["head"]
        for s in ("L", "R"):
            if "eye." + s in self.rp.B:
                self.rp.set_world(pose, "eye." + s, Dh @ q_euler(pitch, 0, yaw))

    def clavicles(self, pose, up_l=0.0, up_r=0.0, fwd=0.0):
        """shrug (deg, + raises the shoulder) and protract (+ rolls shoulders forward)."""
        for s, up in (("L", up_l), ("R", up_r)):
            sg = 1 if s == "L" else -1
            pose["R"]["clavicle." + s] = q_euler(0, -sg * up, -sg * fwd * 0.6)

    def hand_frame(self, side, finger_dir, palm_normal):
        """world wrist rotation that points the fingers along finger_dir with the palm facing palm_normal."""
        hr = self.rp.h_rest
        W, K = hr["wrist." + side], hr["finger3-1." + side]
        I, P = hr["finger2-1." + side], hr["finger5-1." + side]
        a1 = (K - W).normalized()
        n1 = (I - P).cross(a1).normalized()
        if n1.z > 0:
            n1 = -n1
        return frame_rot(a1, n1, Vector(finger_dir), Vector(palm_normal))

    def arm_ik_twist(self, pose, side, target, pole, hand_rot=None, amount=1.0):
        """arm IK; the hand roll is shared with the forearm twist bone (no candy-wrapper wrist)."""
        rp = self.rp
        rp.two_bone(pose, f"upperarm01.{side}", f"upperarm02.{side}", f"lowerarm01.{side}",
                    f"lowerarm02.{side}", f"wrist.{side}", target, pole)
        if hand_rot is None or amount <= 0:
            return
        D2 = pose["D"][f"lowerarm01.{side}"]
        Dw = D2.slerp(hand_rot, amount)
        rel = D2.inverted() @ Dw
        hr = rp.h_rest
        ax = (hr[f"wrist.{side}"] - hr[f"lowerarm01.{side}"]).normalized()
        proj = Vector((rel.x, rel.y, rel.z)).dot(ax)
        tw = Quaternion((rel.w, ax.x * proj, ax.y * proj, ax.z * proj))
        if tw.magnitude < 1e-8:
            tw = Quaternion()
        tw.normalize()
        rp.set_world(pose, f"lowerarm02.{side}", D2 @ Quaternion().slerp(tw, 0.55))
        rp.set_world(pose, f"wrist.{side}", Dw)

    def finish(self, frame, pose):
        self.rp.fk(pose)
        self.rp.store(frame, pose)
        if "SK" in pose:
            self.sk_frames[frame] = dict(pose["SK"])

    def bake_sk(self, objs):
        """write the per-frame expression units as shape-key f-curves on every object that has them."""
        import bpy
        from rigpose import bpy_new_action
        if not self.sk_frames:
            return
        frames = sorted(self.sk_frames)
        names = sorted({k for f in frames for k in self.sk_frames[f]})
        for ob in objs:
            key = ob.data.shape_keys
            if key is None:
                continue
            if key.animation_data is None:
                key.animation_data_create()
            act = bpy_new_action(ob.name + "_expr")
            for nm in names:
                if nm not in key.key_blocks:
                    continue
                fc = act.fcurves.new(f'key_blocks["{nm}"].value')
                vals = [self.sk_frames[f].get(nm, 0.0) for f in frames]
                fc.keyframe_points.add(len(frames))
                co = np.empty(2 * len(frames), dtype=np.float32)
                co[0::2] = frames
                co[1::2] = vals
                fc.keyframe_points.foreach_set("co", co)
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
                fc.update()
            key.animation_data.action = act


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
    T_LEAP = 13.44
    T_IMPACT = 13.62
    T_LAND = 14.20

    def __init__(self, zombie, victim, voice=None):
        self.voice = voice
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
        aL = self.V.feet["L"].eval(self.T_IMPACT)[0]
        aR = self.V.feet["R"].eval(self.T_IMPACT)[0]
        self.v_feet_imp = (aL + aR) * 0.5
        self.vy_ground = self.v_feet_imp.y - 0.62 - (self.V.hip_h - 0.06) - 0.28

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
            fr = 1.62 + 0.66 * clamp(vel / self.VMAX)
            # short ground contacts at speed (~0.085 s at top speed)
            beta = 0.50 - 0.31 * clamp(vel / self.VMAX)
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
                    # foot strikes under / slightly behind the hips: far behind while
                    # accelerating (pushing), almost under the pelvis at top speed
                    kk = clamp(self.z_speed(tm) / self.VMAX)
                    y = self.z_pelvis_y(tm) + lerp(0.24, 0.05, kk)
                    x = 0.085 if s == "L" else -0.085
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
        run = [p for s in "LR" for p in F[s].plants if p.t_on >= self.T_LAUNCH - 0.5]
        self.z_mids = sorted(0.5 * (p.t_on + p.t_off) for p in run)
        self.z_tdL = sorted(p.t_on for p in F["L"].plants if p.t_on >= self.T_LAUNCH - 0.5)
        # sharp nervous jerks of the head/shoulders while running
        self.z_run_twitch = [7.55, 8.62, 9.48, 10.71, 11.83, 12.66]

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
            if self.T_NOTICE <= t < self.T_IMPACT:
                # the zombie's eyes lock onto the prey
                self.Z.eyes_at(zpose, vpose["H"]["head"] + Vector((0, 0, 0.08)), 1.0, limit=32)
            self.V.finish(f, vpose)
            self.Z.finish(f, zpose)
            self.record(f, t, zpose, vpose)
        self.Z.rp.bake("zombie_anim")
        self.V.rp.bake("victim_anim")
        self.V.bake_sk([self.vc.body] + [ob for ob, _ in getattr(self.vc, "proxies", [])])

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
            z_tup=tuple(self.Z.rp.point(zp, "head", self.zc.teeth_up)),
            z_tlo=tuple(self.Z.rp.point(zp, "jaw", self.zc.teeth_lo)),
            z_neck=tuple(zp["H"]["neck02"]), z_spine1=tuple(zp["H"]["spine01"]),
            v_neck_D=tuple(vp["D"]["neck01"]), v_chest=tuple(vp["H"]["spine01"]),
            info=dict(getattr(self, "_z_info", {})) if t >= self.T_IMPACT else {},
        )

    # ===================== ZOMBIE =====================
    def zombie(self, t, vpose):
        b0, b1 = self.T_LAUNCH - 0.06, self.T_LAUNCH + 0.16
        if t < b0:
            return self.z_idle_turn(t)
        if t < b1:
            return self._blend_pose(self.z_idle_turn(t), self.z_sprint(t), sstep(b0, b1, t))
        a0, a1 = self.T_IMPACT - 0.05, self.T_IMPACT + 0.07
        if t < a0:
            return self.z_sprint(t)
        if t < a1:
            return self._blend_pose(self.z_sprint(t), self.z_ground(t, vpose), sstep(a0, a1, t))
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

    # ---- sprint helpers ----
    def _gait_phase(self, t):
        """left-foot cycle phase psi in [0,1): 0 at left touchdown (extrapolated at the ends)."""
        td = self.z_tdL
        if t < td[0]:
            per = td[1] - td[0]
            return ((t - td[0]) / per) % 1.0
        for a, b in zip(td[:-1], td[1:]):
            if t < b:
                return (t - a) / (b - a)
        per = td[-1] - td[-2]
        return ((t - td[-1]) / per) % 1.0

    def _bob(self, t, amp):
        """pelvis vertical offset: lowest at mid-stance, highest in flight (smooth)."""
        m = self.z_mids
        if t <= m[0]:
            return -amp * sstep(m[0] - 0.25, m[0], t)
        for a, b in zip(m[:-1], m[1:]):
            if t < b:
                ph = (t - a) / (b - a)
                return -amp * math.cos(2 * math.pi * ph)
        return -amp

    def _sprint_swing(self, s, t):
        """pelvis-relative swing path of an ankle between two sprint plants:
        toe-off -> heel recovery (heel kicks up behind) -> knee drive -> pawing -> strike.
        Returns (ankle, D_toe_off, D_strike, u) or None while the foot is planted."""
        F = self.Z.feet[s]
        P = F.plants
        for p, q in zip(P[:-1], P[1:]):
            if not (p.t_off < t < q.t_on) or q.kind not in ("sprint", "push") or q.t_on < self.T_LAUNCH:
                continue
            Tsw = q.t_on - p.t_off
            u = (t - p.t_off) / Tsw
            a0, D0, _ = F._stance(p, 1.0)
            a1, D1, _ = F._stance(q, 0.0)
            y0, y1, yp = self.z_pelvis_y(p.t_off), self.z_pelvis_y(q.t_on), self.z_pelvis_y(t)
            f0, f1 = -(a0.y - y0), -(a1.y - y1)          # forward of the pelvis (+)
            v0, v1 = self.z_speed(p.t_off), self.z_speed(q.t_on)
            k = clamp(self.z_speed(0.5 * (p.t_off + q.t_on)) / self.VMAX)
            pts_f = [(0.0, f0),
                     (0.24, lerp(f0 + 0.10, -0.60, k)),
                     (0.46, lerp(-0.12, -0.30, k)),
                     (0.68, lerp(0.10, 0.16, k)),
                     (0.86, f1 + lerp(0.06, 0.12, k)),
                     (1.0, f1)]
            pts_z = [(0.0, a0.z),
                     (0.24, lerp(0.24, 0.42, k)),
                     (0.46, lerp(0.30, 0.55, k)),
                     (0.68, lerp(0.30, 0.46, k)),
                     (0.86, lerp(0.16, 0.22, k)),
                     (1.0, a1.z)]
            fr = spline(u, pts_f, m0=-v0 * Tsw * 0.55, m1=-v1 * Tsw * 0.35)
            zz = spline(u, pts_z, m0=0.9, m1=-0.9)
            x = lerp(a0.x, a1.x, sstep(0.0, 1.0, u))
            return Vector((x, yp - fr, max(zz, 0.07))), D0, D1, u
        return None

    def _sprint_leg(self, pose, s, ankle, D0, D1, u):
        """IK a swinging leg; the foot hangs off the shank (plantar-flexed after push-off,
        toes pulled up for the strike), blended into the plant rotations at both ends."""
        rp = self.Z.rp
        fwd = Vector((0, -1, 0))
        sgn = 1 if s == "L" else -1
        pole = fwd + Vector((sgn * 0.12, 0, 0))
        rp.two_bone(pose, f"upperleg01.{s}", f"upperleg02.{s}", f"lowerleg01.{s}", f"lowerleg02.{s}",
                    f"foot.{s}", ankle, pole)
        D2 = pose["D"][f"lowerleg02.{s}"]
        plantar = spline(u, [(0.0, 38), (0.25, 34), (0.48, 18), (0.68, -6), (0.86, -10), (1.0, 2)])
        Df = D2 @ q_euler(plantar, 0, 0)
        Df = D0.slerp(Df, sstep(0.0, 0.2, u))
        Df = Df.slerp(D1, sstep(0.84, 1.0, u))
        rp.set_world(pose, f"foot.{s}", Df)
        tb = -lerp(30, 0, sstep(0.0, 0.4, u))
        for toe in (1, 2, 3, 4, 5):
            pose["R"][f"toe{toe}-1.{s}"] = q_euler(tb, 0, 0)

    def _sprint_arm(self, pose, s, alpha, flex, abd, cross):
        """angle-driven arm: alpha shoulder swing (+ forward) and abd (out) relative to the
        trunk, flex = elbow bend; hands drift toward the midline (cross) in front."""
        Z = self.Z
        rp = Z.rp
        f, u, l = Z.chest_frame(pose)
        sg = 1 if s == "L" else -1
        lat = l * sg
        S = pose["H"][f"upperarm01.{s}"]
        L1 = (rp.h_rest[f"lowerarm01.{s}"] - rp.h_rest[f"upperarm01.{s}"]).length
        L2 = (rp.h_rest[f"wrist.{s}"] - rp.h_rest[f"lowerarm01.{s}"]).length
        a, g, b = math.radians(alpha), math.radians(abd), math.radians(flex)
        d_up = (-u) * math.cos(a) + f * math.sin(a)
        d_up = (d_up * math.cos(g) + lat * math.sin(g)).normalized()
        p = f * math.cos(a) + u * math.sin(a)
        p = (p - d_up * p.dot(d_up)).normalized()
        fa = (d_up * math.cos(b) + p * math.sin(b) - lat * (cross * math.sin(b))).normalized()
        E = S + d_up * L1
        H = E + fa * L2
        Z.arm_ik(pose, s, H, E - (S + H) * 0.5)

    def z_sprint(self, t):
        Z, n = self.Z, self.nz
        v = self.z_speed(t)
        k = clamp(v / self.VMAX)
        ka = k ** 0.7
        psi = self._gait_phase(t)
        sw = {s: self._sprint_swing(s, t) for s in ("L", "R")}
        feet = {}
        for s in ("L", "R"):
            if sw[s] is not None:
                feet[s] = sw[s][0]
            else:
                feet[s] = Z.feet[s].eval(t)[0]
        aL, aR = feet["L"], feet["R"]
        y = self.z_pelvis_y(t)
        # pelvis: crouched while driving out of the start, tall and bouncy at speed
        tw = twitch(t, self.z_run_twitch, 0.05, 0.3)
        lat_sway = 0.022 * k * clamp((aR.z - aL.z) / 0.35, -1, 1)
        x = lat_sway + 0.008 * n(t, 1.3)
        base = lerp(0.885, 0.99, ka)
        z = base + self._bob(t, lerp(0.014, 0.032, k))
        leap = 0.0
        if t > self.T_LEAP:
            u = (t - self.T_LEAP) / (self.T_IMPACT - self.T_LEAP)
            leap = 0.12 * math.sin(math.pi * clamp(u))
        pos = Vector((x, y, z + leap))
        pitch = lerp(40, 17, ka) + 3 * n(t, 1.1) + sstep(self.T_LEAP, self.T_IMPACT, t) * 24
        yaw = -clamp((aR.y - aL.y) / 1.0, -1, 1) * 10 * k
        roll = 3.5 * k * clamp((aL.z - aR.z) / 0.35, -1, 1)
        rq = yawq(yaw) @ q_euler(pitch, roll, 0)
        pose = Z.begin(pos, rq)
        Z.spine(pose, bend=lerp(14, 9, k) + 3 * n(t, 0.9), side=-roll * 0.9 + 7 * tw * n(t, 3.0),
                twist=-yaw * 2.3 + 4 * n(t, 1.7))
        Z.rp.fk(pose)
        # head locked onto the prey, jutting forward; twitchy rolls
        vic = Vector((0, self.v_pelvis_y(t), 1.5))
        Z.look_at(pose, vic, amount=1.0, roll=5 * n(t, 1.6) + 16 * tw, extra_pitch=-6)
        # arms: big, loose, clawing swing opposite to the legs (left arm forward when the
        # right leg drives forward), elbows opening as the claws reach for the prey
        leap_k = sstep(self.T_LEAP - 0.12, self.T_LEAP + 0.1, t)
        amp = lerp(42, 62, k)
        mid = lerp(8, 14, k)
        for s, ph0 in (("L", 0.45), ("R", 0.95)):
            sg = 1 if s == "L" else -1
            c = math.cos(2 * math.pi * (psi - ph0))
            alpha = mid + amp * c + 7 * math.sin(4 * math.pi * (psi - ph0)) + 6 * n(t + 4 * sg, 1.4)
            front = clamp((alpha + 40) / 120)
            flex = lerp(88, 52, front) + 6 * n(t + 2 * sg, 1.9)
            abd = 15 + 6 * front + 5 * n(t + 7 * sg, 0.8) + (5 if s == "R" else 0) + 14 * tw * (sg > 0)
            cross = 0.28 * front
            alpha = lerp(alpha, 92, leap_k)
            flex = lerp(flex, 28, leap_k)
            abd = lerp(abd, 24, leap_k)
            cross = lerp(cross, 0.0, leap_k)
            self._sprint_arm(pose, s, alpha, flex, abd, cross)
            Z.fingers(pose, s, curl=lerp(0.62, 0.35, leap_k) + 0.12 * n(t + sg, 3), spread=0.9, thumb=0.5)
        # legs: planted feet via the plant tracks, swinging feet via the sprint path
        ov = {}
        for s_, sg in (("L", 1), ("R", -1)):
            lp = self.z_last[s_]
            if t > lp.t_off:
                u = sstep(lp.t_off, lp.t_off + 0.22, t)
                hip = pose["H"]["upperleg01." + s_]
                a_end = hip + Vector((sg * 0.12, 0.72 + 0.1 * sg, -0.36 + 0.14 * sg))
                a0, D0, p0, _ = Z.feet[s_].eval(lp.t_off)
                ov[s_] = (a0.lerp(a_end, u), q_euler(lerp(p0, 65, u), 0, 0))
        Z.rp.fk(pose)
        stance = [s for s in ("L", "R") if s not in ov and sw[s] is None]
        if stance or ov:
            Z.legs_ik(pose, t, override=ov or None)
        for s in ("L", "R"):
            if s not in ov and sw[s] is not None:
                Z.rp.fk(pose)
                self._sprint_leg(pose, s, *sw[s])
        jaw = 22 + 5 * math.sin(t * 9.0) + 4 * n(t, 3) + 6 * tw
        Z.face(pose, jaw=jaw, snarl=1.0, lowlip=0.85, lids=-0.2, brow=-1.0)
        return pose

    # ---- feeding ----
    # (lunge_start, clamp, tear_start, tear_end, rip, chew_end)
    BITES = [(13.62, 14.30, 15.55, 16.70, 16.90, 21.15),
             (21.20, 21.60, 22.60, 24.10, 24.35, 31.0)]
    SWALLOWS = [21.05, 28.6]
    T_SHOCK = 16.95

    def bite_params(self, t):
        """b: 0 head raised .. 1 mouth on the wound; pull: tearing back; shake: head yaw (deg)."""
        b = pull = shake = chew_on = 0.0
        for i, (l0, cl, t0, t1, rip, ce) in enumerate(self.BITES):
            if l0 <= t < rip + 0.3:
                if t < cl:
                    b = max(b, 0.86 + 0.14 * sstep(self.T_LAND, cl, t) if i == 0 else sstep(l0, cl, t))
                elif t < t1:
                    b = 1.0
                elif t < rip:
                    b = lerp(1.0, 0.8, sstep(t1, rip, t))
                else:
                    b = max(b, lerp(0.8, 0.0, sstep(rip, rip + 0.25, t)))
                if cl <= t < t0:      # biting in, worrying the wound
                    env = sstep(cl, cl + 0.15, t) * (1 - sstep(t0 - 0.12, t0, t))
                    shake += 6 * env * math.sin((t - cl) * 2 * math.pi * 5.5)
                if t0 <= t < rip:     # violent tearing
                    env = sstep(t0, t0 + 0.12, t) * (1 - sstep(rip - 0.06, rip, t))
                    shake += 19 * env * math.sin((t - t0) * 2 * math.pi * 4.2)
                if t1 - 0.45 <= t < rip + 0.3:
                    pull = max(pull, sstep(t1 - 0.45, rip, t) * (1 - sstep(rip + 0.05, rip + 0.3, t)))
            if rip + 0.12 <= t < ce:
                chew_on = max(chew_on, sstep(rip + 0.12, rip + 0.45, t) * (1 - sstep(ce - 0.2, ce, t)))
        return b, pull, shake, chew_on

    def v_frame(self, vpose):
        D = vpose["D"]["root"]
        return (vpose["H"]["root"], D @ Vector((0, -1, 0)), D @ Vector((0, 0, 1)), D @ Vector((1, 0, 0)))

    def _zombie_upper(self, t, P, M, G, roll, bend, yaw, extra_pitch=0.0, max_fix=0.16):
        """place pelvis at P, find torso pitch so the mouth reaches M while the face looks at G."""
        Z = self.Z
        rp = Z.rp
        mouth_rest = self.zc.mouth

        def build(theta):
            pose = Z.begin(P, yawq(yaw) @ q_euler(theta, roll * 0.15, 0))
            Z.spine(pose, bend=bend, side=roll * 0.1, twist=-yaw * 0.6)
            rp.fk(pose)
            Z.look_at(pose, G, amount=1.0, up=None, roll=roll, extra_pitch=extra_pitch)
            return pose, rp.point(pose, "jaw", mouth_rest)

        # temporal coherence: stay near last frame's solution (no flips between minima)
        prev = getattr(self, "_zu_prev", None)
        coherent = prev is not None and 0.0 < t - prev[0] < 0.05
        lo, hi = (max(0.0, prev[1] - 9.0), min(150.0, prev[1] + 9.0)) if coherent else (0.0, 150.0)
        gr = 0.6180339887
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = (build(c)[1] - M).length
        fd = (build(d)[1] - M).length
        for _ in range(16):
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = (build(c)[1] - M).length
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = (build(d)[1] - M).length
        theta = 0.5 * (lo + hi)
        pose, mouth = build(theta)
        fix = M - mouth
        if fix.length > max_fix:
            fix = fix.normalized() * max_fix
        if coherent:
            dfix = fix - prev[2]
            if dfix.length > 0.02:
                fix = prev[2] + dfix.normalized() * 0.02
        self._zu_prev = (t, theta, fix.copy())
        pose["root_loc"] += fix
        rp.fk(pose)
        self._z_theta = theta
        return pose

    def _blend_pose(self, pa, pb, w):
        out = self.Z.rp.new_pose()
        out["root_loc"] = pa["root_loc"].lerp(pb["root_loc"], w)
        for k in set(pa["R"]) | set(pb["R"]):
            qa = pa["R"].get(k, Quaternion())
            qb = pb["R"].get(k, Quaternion())
            out["R"][k] = qa.slerp(qb, w)
        self.Z.rp.fk(out)
        return out

    def z_ground(self, t, vpose):
        Z, V, n = self.Z, self.V, self.nz
        rp = Z.rp
        T0, T1 = self.T_IMPACT, self.T_LAND
        vroot, vfront, vaxis, vlat = self.v_frame(vpose)
        bite = V.rp.point(vpose, "neck01", self.vc.bite_point)
        b, pull, shake, chew_on = self.bite_params(t)
        falling = t < T1 + 0.05
        uf = sstep(T0, T1, t)
        breath = math.sin(t * 2 * math.pi * 0.45)
        swallow = twitch(t, self.SWALLOWS, 0.12, 0.35)
        # ---- targets in the victim's body frame
        if falling:
            P = vroot + vfront * lerp(0.42, 0.41, uf) + vaxis * lerp(-0.08, 0.07, uf)
            M = bite + vfront * lerp(0.12, 0.05, uf)
            G = bite - vfront * 0.55   # far target: stable gaze while the head passes the wound
            roll = -22
        else:
            P_bite = vroot + vfront * 0.41 + vaxis * 0.08
            P_chew = vroot + vfront * 0.31 - vaxis * 0.07
            P = P_chew.lerp(P_bite, b) + vfront * (0.04 * pull) - vaxis * (0.06 * pull)
            P += vfront * (0.012 * breath) + vlat * (0.02 * n(t, 0.3))
            M_bite = bite + vfront * 0.012
            M_hover = bite + vfront * 0.30 - vaxis * 0.17 + vlat * 0.02 + vfront * (0.02 * n(t, 0.5))
            M = M_hover.lerp(M_bite, b) + vfront * (0.07 * pull) - vaxis * (0.05 * pull)
            M += vfront * (0.05 * swallow) - vaxis * (0.03 * swallow)
            G_bite = bite - vfront * 0.5
            # chewing: head up, gaze forward along the street, slowly scanning
            G_chew = M_hover + vaxis * 0.6 - vfront * 0.10 + vlat * (0.18 * n(t, 0.18))
            G = G_chew.lerp(G_bite, b)
            roll = lerp(-6, -30, b)
        bend = lerp(8, 26, b) + 2 * breath - 10 * pull
        yaw = 4 * n(t, 0.3) + 0.22 * shake
        pose = self._zombie_upper(t, P, M, G, roll, bend, yaw, extra_pitch=-8 * swallow)
        # ---- tearing: head worries the flesh (shake about the neck axis)
        if abs(shake) > 1e-3:
            for bn, w in (("neck02", 0.3), ("neck03", 0.3), ("head", 0.4)):
                pose["R"][bn] = pose["R"][bn] @ q_euler(0, 0.25 * shake * w, shake * w)
            rp.fk(pose)
        # ---- legs: kneeling astride the victim's hips, shins flat behind
        L1 = (rp.h_rest["lowerleg01.L"] - rp.h_rest["upperleg01.L"]).length
        s_land = sstep(T0 + 0.12, T1 + 0.1, t)
        Droot = pose["D"]["root"]
        for s, sg in (("L", 1), ("R", -1)):
            hip = pose["H"]["upperleg01." + s]
            a_lat = 0.13
            dz = hip.z - 0.07
            b2 = L1 * L1 - a_lat * a_lat - dz * dz
            if b2 > 0:
                Kg = Vector((hip.x + sg * a_lat, hip.y - math.sqrt(b2), 0.07))
            else:
                Kg = hip + Vector((sg * a_lat, -0.04, -dz)).normalized() * L1
            Kg.z += 0.03 * max(0.0, n(t + 3 * sg, 0.35))       # shifting weight
            Ag = Kg + Vector((sg * 0.05, 0.44, 0.0))
            Ag.z = 0.085
            if s_land < 1.0:
                Ka = hip + Droot @ Vector((sg * 0.10, 0.12, -0.42))
                Aa = Ka + Droot @ Vector((sg * 0.03, 0.36, 0.18))
                Kg = Ka.lerp(Kg, s_land)
                Ag = Aa.lerp(Ag, s_land)
            Dfoot = q_euler(lerp(70, 105, s_land), 0, 0)
            rp.limb_points(pose, f"upperleg01.{s}", f"upperleg02.{s}", f"lowerleg01.{s}", f"lowerleg02.{s}",
                           f"foot.{s}", Kg, Ag, Vector((0, -1, 0.4)), end_rot=Dfoot)
            for toe in (1, 2, 3, 4, 5):
                pose["R"][f"toe{toe}-1.{s}"] = q_euler(-30 * s_land, 0, 0)
        rp.fk(pose)
        # ---- arms
        vsh = {s: V.rp.point(vpose, "upperarm01." + s, V.rp.h_rest["upperarm01." + s]) for s in ("L", "R")}
        vchest = vpose["H"]["spine01"]
        vhead = vpose["H"]["head"] + vaxis * 0.07
        vuparm_L = vpose["H"]["upperarm02.L"]
        out_w = 1 - sstep(T1 - 0.05, T1 + 0.3, t)          # arms come out from under the victim
        for s, sg in (("L", 1), ("R", -1)):
            other = "R" if s == "L" else "L"                   # face to face: mirror sides
            wrap = vsh[other] - vfront * 0.10 - vaxis * 0.04
            pin = vsh[other] + vfront * 0.075 + vaxis * 0.01
            if s == "L":
                bite_t = pin
                tear_t = pin + vaxis * 0.02
                chew_t = vchest + vfront * 0.11 - vlat * 0.06 - vaxis * 0.05
            else:
                bite_t = vhead + vlat * 0.075 + vfront * 0.05       # pushes the head aside
                tear_t = vchest + vlat * 0.13 + vfront * 0.10
                chew_t = vuparm_L + vfront * 0.055
            ground_t = bite_t.lerp(tear_t, pull)
            ground_t = chew_t.lerp(ground_t, clamp(b + pull))
            ground_t += Vector((0.012 * n(t + 5 * sg, 2.0), 0.012 * n(t + 9 * sg, 2.2), 0.0))
            tgt = ground_t.lerp(wrap, out_w)
            f, u, l = Z.chest_frame(pose)
            pole = l * sg * 1.0 + Vector((0, 0, 0.9)) - f * 0.3
            Z.arm_ik(pose, s, tgt, pole)
            Z.fingers(pose, s, curl=0.7 + 0.2 * pull + 0.1 * n(t + sg * 2, 2.0), spread=0.6, thumb=0.6)
        # ---- face
        l0, cl = self.BITES[0][0], self.BITES[0][1]
        jaw_open = 0.0
        for (l0, cl, t0, t1, rip, ce) in self.BITES:
            if l0 <= t < cl:
                jaw_open = max(jaw_open, sstep(l0, l0 + 0.15, t) * (1 - sstep(cl - 0.06, cl, t)))
        cph = (t * 1.55) % 1.0
        chew = sstep(0, 0.6, cph) if cph < 0.6 else 1 - sstep(0.6, 1.0, cph) ** 0.7
        chew_amp = 11 + 4 * n(t, 0.3)
        if b > 0.5:
            jaw = lerp(4 + 2 * abs(math.sin(t * 13)), 27, jaw_open)
            snarl = 1.0
        else:
            jaw = chew_on * (3 + chew_amp * chew) + (1 - chew_on) * 7 + 18 * (1 - chew_on) * (1 - b) * sstep(14.5, 15.0, t) * 0
            snarl = 0.5 + 0.5 * (1 - chew_on) + 0.2 * chew
        jaw *= (1 - 0.8 * swallow)
        grind = chew_on * 3.5 * math.sin(t * 1.55 * 2 * math.pi + 1.2) * (1 if int(t * 1.55) % 2 else -1)
        Z.face(pose, jaw=jaw, snarl=snarl, lowlip=0.6 + 0.3 * chew_on * chew, jaw_side=grind,
               lids=0.25 * chew_on, brow=-0.8)
        self._z_info = dict(b=b, chew=chew * chew_on, shake=shake, pull=pull)
        return pose

    # ===================== VICTIM =====================
    T_SEAT = 14.06        # buttocks hit the asphalt (back follows at T_LAND)

    def _voice_at(self, t):
        """victim voice: (envelope 0..1, kind) at time t, from voice.json (lip sync / effort)."""
        vo = self.voice
        if not vo:
            k = "scream" if (12.92 < t < 13.64 or 14.36 < t < 16.78 or 16.8 < t < 17.35) else "none"
            return (0.7 if k == "scream" else 0.0), k
        x = t * FPS
        i = int(x)
        env = vo["env"]
        a = env[min(max(i, 0), len(env) - 1)]
        b = env[min(max(i + 1, 0), len(env) - 1)]
        k = vo["kind"][min(max(int(round(x)), 0), len(env) - 1)]
        return lerp(a, b, x - i), k

    def _prep_voice(self):
        """effort envelope + struggle bursts derived from the voice, so the body fights when he screams."""
        TL, cl, rip = self.T_LAND, self.BITES[0][1], self.BITES[0][4]
        fr = np.arange(NFRAMES) / FPS
        env = np.array([self._voice_at(t)[0] for t in fr])
        eff = np.zeros_like(env)
        for i in range(1, len(env)):
            k = 0.55 if env[i] > eff[i - 1] else 0.12
            eff[i] = eff[i - 1] + k * (env[i] - eff[i - 1])
        self.v_eff = eff
        import voice as _voice
        self.v_bursts = _voice.bursts(list(env), {"T_LAND": TL, "BITES": self.BITES})

    def _burst(self, t, dur=0.42):
        """(strength 0..1, lead side +-1, phase) of the struggle burst active at t."""
        best, side, ph = 0.0, 1, 0.0
        for (tb, sd, a) in getattr(self, "v_bursts", []):
            x = (t - tb + 0.06) / dur
            if 0.0 <= x < 1.6:
                v = a * (sstep(0.0, 0.25, x) * (1 - sstep(0.55, 1.6, x)))
                if v > best:
                    best, side, ph = v, sd, x
        return min(1.0, best), side, ph

    def victim_body(self, t):
        if not hasattr(self, "v_bursts"):
            self._prep_voice()
        if t < self.T_IMPACT:
            pose = self.v_standing(t)
        else:
            pose = self.v_fall_ground(t)
        self.v_face(pose, t)
        return pose

    # ---------------- standing: walking on his phone, startle, freeze, backpedal
    def _v_stand_params(self, t):
        TN = self.T_V_NOTICE
        n = self.nv
        walking = 1 - sstep(TN, self.T_V_STOP, t)
        jolt = twitch(t, [TN + 0.03], 0.08, 0.5)
        fear = sstep(TN, TN + 0.2, t)
        back = sstep(12.78, 13.45, t)
        brace = sstep(13.36, 13.58, t)
        breath = math.sin(2 * math.pi * 2.3 * (t - TN)) * fear * (1 - 0.6 * back)
        trem = fear * (0.6 * n(t, 7.0) + 0.4 * n(t + 3.0, 11.0))
        return walking, jolt, fear, back, brace, breath, trem

    def _v_stand_root(self, t):
        V = self.V
        aL = V.feet["L"].eval(t)[0]
        aR = V.feet["R"].eval(t)[0]
        walking, jolt, fear, back, brace, breath, trem = self._v_stand_params(t)
        mid = (aL + aR) * 0.5
        dy = abs(aL.y - aR.y)
        bob = 0.025 * clamp(1 - dy / 0.7) * walking
        ph = t * 2 * math.pi * 0.9
        crouch = 0.035 * jolt + 0.05 * back + 0.04 * brace
        pos = Vector((mid.x * 0.6 + 0.015 * math.sin(ph) * walking, self.v_pelvis_y(t), V.hip_h - 0.03 - bob - crouch))
        pitch = 4 * walking + 3 * jolt - 8 * back - 5 * brace
        yaw = 180 + 5 * math.sin(ph) * walking + 12 * back + 10 * brace
        roll = 2 * math.sin(ph) * walking
        return pos, yaw, pitch, roll

    def v_standing(self, t):
        V, n = self.V, self.nv
        TN = self.T_V_NOTICE
        walking, jolt, fear, back, brace, breath, trem = self._v_stand_params(t)
        pos, yaw, pitch, roll = self._v_stand_root(t)
        pose = V.begin(pos, yawq(yaw) @ q_euler(pitch, roll, 0))
        ph = t * 2 * math.pi * 0.9
        V.spine(pose, bend=6 * walking + 8 * jolt + 6 * back + 10 * brace + 1.4 * breath,
                side=0.8 * trem, twist=-5 * math.sin(ph) * walking - 10 * back - 8 * brace)
        shrug = 13 * jolt + 6 * fear + 9 * brace + 2.5 * breath
        V.clavicles(pose, shrug, shrug, fwd=5 * back + 8 * brace)
        V.rp.fk(pose)
        zh = Vector((0.0, self.z_pelvis_y(t) - 0.25, 1.30))
        if t < TN:
            V.neck_head(pose, pitch=28 + 2 * n(t, 0.5), yaw=3 * n(t, 0.3), roll=2)
        else:
            look = sstep(TN, TN + 0.13, t)
            V.neck_head(pose, pitch=28 * (1 - look), yaw=3 * (1 - look), roll=2)
            V.look_at(pose, zh, amount=look, extra_pitch=-3 * look)
            # head draws back and turns away from the blow while the eyes stay on it
            away = 8 * back + 26 * brace
            for b, w in (("neck02", 0.3), ("neck03", 0.3), ("head", 0.4)):
                pose["R"][b] = pose["R"][b] @ q_euler((-6 * jolt + 6 * brace + 1.5 * trem) * w, -4 * brace * w, away * w)
            V.rp.fk(pose)
        self._v_arm_state = (walking, jolt, fear, back, brace, breath, trem)
        self._v_look = zh
        V.legs_ik(pose, t)
        return pose

    # ---------------- knocked down: hips first, then the back, then the head
    def _v_lie(self, t):
        """root position / lying angle for the fall and the ground (no struggle terms)."""
        T0, TS, TL = self.T_IMPACT, self.T_SEAT, self.T_LAND
        p0, yaw0, pitch0, roll0 = self._v_stand_root(T0)
        a0 = -pitch0
        w = clamp((t - T0) / (TS - T0))
        u = clamp((t - T0) / (TL - T0))
        alpha = a0 + (90.0 - a0) * (0.16 * u + 0.84 * u ** 2.3)
        y_seat = self.vy_ground + 0.20
        if t <= TS:
            y = p0.y + (y_seat - p0.y) * (1 - (1 - w) ** 1.7)
            z = p0.z - (p0.z - 0.14) * w ** 2.0
        else:
            dt_ = t - TS
            y = y_seat + (self.vy_ground - y_seat) * (1 - math.exp(-dt_ * 6.5))
            z = lerp(0.14, 0.12, sstep(TS, TL + 0.1, t)) + 0.03 * math.exp(-dt_ * 11) * abs(math.sin(dt_ * 19))
        if t > TL:
            dt_ = t - TL
            alpha += -5 * math.exp(-dt_ * 9) * math.sin(dt_ * 24)   # the back bounces on the asphalt
        yaw = lerp(yaw0, 180.0, sstep(T0, TL, t))
        x = lerp(p0.x, 0.0, sstep(T0, TL, t))
        return Vector((x, y, z)), yaw, alpha

    def v_fall_ground(self, t):
        V, n = self.V, self.nv
        T0, TS, TL = self.T_IMPACT, self.T_SEAT, self.T_LAND
        l0, cl, t0, t1, rip, ce = self.BITES[0]
        TSH = self.T_SHOCK
        pos, yaw, alpha = self._v_lie(t)
        # ---- how much fight is left
        fight = keys(t, [(TL, 0.75), (cl, 1.0), (rip - 0.05, 1.0), (TSH + 0.2, 0.25), (TSH + 1.2, 0.08),
                         (21.4, 0.04), (21.75, 0.22), (22.6, 0.08), (24.4, 0.0)])
        bst, bside, bph = self._burst(t)
        bst *= clamp(fight * 1.15)
        eff = float(self.v_eff[min(NFRAMES - 1, int(round(t * FPS)))]) * fight
        b, pull, shake, chew_on = self.bite_params(t)
        tear_w = sum(sstep(bt0, bt0 + 0.1, t) * (1 - sstep(brip, brip + 0.12, t)) for (_, _, bt0, _, brip, _) in self.BITES)
        rip_jolt = twitch(t, [rip + 0.02], 0.06, 0.5) + 0.5 * twitch(t, [self.BITES[1][4] + 0.02], 0.06, 0.4)
        clamp_jolt = twitch(t, [cl + 0.01], 0.06, 0.45) + 0.4 * twitch(t, [self.BITES[1][1] + 0.01], 0.06, 0.35)
        agonal = twitch(t, [24.75, 25.35, 25.9, 26.8], 0.05, 0.35) * (1 - sstep(27.0, 28.0, t))
        ground = sstep(TL - 0.02, TL + 0.12, t)
        # ---- breathing: panting in the fight, fast shallow in shock, slowing to nothing
        rate = keys(t, [(TL, 1.1), (TSH, 1.2), (TSH + 0.5, 1.6), (21.0, 1.0), (24.3, 0.6), (26.0, 0.25)])
        amp = keys(t, [(TL, 1.0), (TSH, 1.0), (TSH + 0.5, 1.6), (21.0, 1.0), (24.3, 0.6), (26.5, 0.0)])
        self._v_breath_ph = getattr(self, "_v_breath_ph", 0.0)
        bre = amp * math.sin(2 * math.pi * rate * t)
        # ---- writhing / bucking (only on the ground)
        buck = ground * (0.055 * bst * (1 if bph < 0.9 else 0.6) + 0.015 * eff)
        roll = ground * (9 * bst * bside + 3 * fight * n(t, 0.7)) + 4 * rip_jolt
        roll += 3.0 * tear_w * math.sin(2 * math.pi * 4.2 * t + 0.6)          # yanked by the tearing
        alpha_eff = alpha - ground * (7 * pull + 4 * rip_jolt + 2.5 * clamp_jolt)   # shoulders lifted by the pull
        pos = pos + Vector((0.0, 0.0, buck + 0.012 * agonal))
        rq = yawq(yaw + ground * 3 * bside * bst) @ q_euler(-alpha_eff - 1.5, roll, 0)
        pose = V.begin(pos, rq)
        # ---- spine
        u = clamp((t - T0) / (TL - T0))
        fall_bend = keys(t, [(T0, 0.0), (T0 + 0.10, -14.0), (T0 + 0.3, 10.0), (TS, 22.0), (TL, 12.0), (TL + 0.25, 4.0)])
        bend = fall_bend + ground * (-8 * bst + 2.5 * eff - 9 * rip_jolt + 5 * clamp_jolt + 1.6 * bre)
        side = ground * (6 * bst * bside + 4 * fight * n(t + 2, 0.6)) + 2.5 * tear_w * math.sin(2 * math.pi * 4.2 * t)
        twist = ground * (10 * bst * bside + 5 * fight * n(t + 7, 0.5))
        V.spine(pose, bend=bend, side=side, twist=twist)
        V.clavicles(pose, 6 * bst + 4 * rip_jolt + 1.5 * bre, 6 * bst + 4 * rip_jolt + 1.5 * bre, fwd=4 * bst)
        # ---- head: tucked while falling, whips back on the asphalt, then turned away from the bite
        whip = 0.0
        if t > TL:
            dt_ = t - TL - 0.04
            whip = -18 * math.exp(-max(0, dt_) * 7) * math.cos(max(0, dt_) * 24) if dt_ > 0 else 0.0
        tuck = keys(t, [(T0, 0.0), (T0 + 0.12, -8.0), (TS, 30.0), (TL, 26.0), (TL + 0.06, 0.0)])
        turn = sstep(TL, TL + 0.45, t)
        thrash = ground * fight * (14 * bst * bside + 6 * n(t, 1.6))
        dead = sstep(24.4, 26.5, t)
        shock = sstep(TSH, TSH + 0.6, t)
        hp = tuck * (1 - ground) + ground * (-12 + 4 * eff + 6 * shock + 8 * clamp_jolt - 6 * rip_jolt) + whip
        hy = -40 * turn + thrash - 6 * shock + 10 * shock * (1 - dead) * n(t, 0.15) - 14 * dead
        hr = -12 * turn + 4 * bst * bside - 6 * dead
        V.neck_head(pose, pitch=hp, yaw=hy, roll=hr)
        V.rp.fk(pose)
        self._head_on_ground(pose)
        # ---- legs
        Dv = pose["D"]["root"]
        vroot, vfront, vaxis, vlat = self.v_frame(pose)
        ov, kd = {}, {}
        for s, sg in (("L", 1), ("R", -1)):
            hipp = pose["H"]["upperleg01." + s]
            a_stand = V.feet[s].eval(T0)[0]
            # leg positions in the body frame (sg: + = his left)
            bent = vroot - vaxis * 0.50 + vlat * (sg * 0.14)
            straight = vroot - vaxis * 0.76 + vlat * (sg * 0.15)
            limp_l = vroot - vaxis * (0.75 if s == "L" else 0.77) + vlat * (sg * (0.19 if s == "L" else 0.16))
            if t < TS:
                # dragged with the body, then the feet leave the ground
                w = clamp((t - T0) / (TS - T0))
                drag = Vector((0.0, (pos.y - self._v_stand_root(T0)[0].y) * 0.45, 0.0))
                a = (a_stand + drag).lerp(bent, sstep(0.25, 1.0, w))
                a.z = max(0.08, a.z) + 0.10 * sstep(0.5, 1.0, w)
                D = yawq(180) @ q_euler(lerp(0, 40, w), 0, 0)
            else:
                # legs fly up as the back slams down, then the fight
                fly = math.sin(math.pi * clamp((t - TS) / 0.42)) * (1 - sstep(TL + 0.3, TL + 0.4, t))
                kick_on = bst * (1 if bside == sg else 0.35)
                kick = sstep(0.0, 0.35, bph) * (1 - sstep(0.5, 1.2, bph)) * kick_on
                scrape = math.sin(math.pi * clamp(bph / 1.2)) * kick_on
                base = bent.lerp(straight, clamp(0.12 + 0.75 * (1 - fight) + 0.15 * sg * n(t + sg, 0.4)))
                base = base.lerp(limp_l, dead)
                a = base - vaxis * (0.13 * kick) + vfront * (0.12 * kick)
                a += vlat * (sg * 0.04 * scrape) - vaxis * (0.05 * scrape)
                a.z = max(0.075, min(a.z, 0.08 + 0.32 * fly + 0.22 * kick)) if fly < 0.02 else 0.08 + 0.30 * fly
                a.z += 0.02 * agonal * (1 if s == "R" else 0)
                plant = yawq(180 - sg * 10) @ q_euler(0, 0, 0)
                relax = Dv @ q_euler(42, 0, sg * 38)
                D = plant.slerp(relax, clamp(0.25 + 0.55 * (1 - fight) + 0.4 * kick + 0.5 * fly))
            ov[s] = (a, D)
            out = vlat * sg
            kd[s] = vfront * 1.0 + out * lerp(0.35, 0.9 if s == "L" else 0.25, dead)
        V.legs_ik(pose, t, knee_dir=kd, override=ov)
        self._v_ground = dict(fight=fight, bst=bst, bside=bside, bph=bph, eff=eff, pull=pull, b=b,
                              rip_jolt=rip_jolt, clamp_jolt=clamp_jolt, dead=dead, shock=shock,
                              agonal=agonal, bre=bre, tear_w=tear_w, ground=ground)
        return pose

    def _head_on_ground(self, pose):
        """keep the back of the skull on (not in) the asphalt."""
        V = self.V
        rp = V.rp
        for _ in range(3):
            back = rp.point(pose, "head", rp.h_rest["head"] + Vector((0.0, 0.085, 0.09)))
            top = rp.point(pose, "head", rp.h_rest["head"] + Vector((0.0, 0.0, 0.19)))
            low = min(back.z, top.z)
            if low >= 0.075:
                break
            fix = min(25.0, (0.075 - low) * 260)
            for b, w in (("neck01", 0.3), ("neck02", 0.3), ("head", 0.4)):
                pose["R"][b] = pose["R"][b] @ q_euler(fix * w, 0, 0)
            rp.fk(pose)

    # ---------------- face: expression units + jaw from the voice
    def v_face(self, pose, t):
        V, n = self.V, self.nv
        env, kind = self._voice_at(t)
        TN, TI, TL = self.T_V_NOTICE, self.T_IMPACT, self.T_LAND
        TSH = self.T_SHOCK
        fear = sstep(TN, TN + 0.15, t) * (1 - sstep(TI, TI + 0.2, t))
        brace = sstep(13.40, 13.58, t) * (1 - sstep(TI + 0.1, TI + 0.3, t))
        fight = sstep(TI, TI + 0.1, t) * (1 - sstep(TSH, TSH + 0.3, t))
        shock = sstep(TSH, TSH + 0.25, t)
        dead = sstep(24.4, 26.5, t)
        g = getattr(self, "_v_ground", {}) if t >= TI else {}
        voc = clamp(env * 1.5)
        loud = kind in ("scream", "gasp", "grunt", "moan") and env > 0.05
        # semantic channels
        scream = (voc if loud else 0.0) * (1 - dead)
        pain = (0.55 * fight + 0.35 * g.get("bst", 0.0)) * (1 - 0.6 * scream)
        pain += 0.6 * g.get("clamp_jolt", 0.0) + 0.8 * g.get("rip_jolt", 0.0) * (1 - dead)
        wide = 0.9 * fear + shock * (0.8 - 0.6 * sstep(TSH + 1.0, 23.0, t)) * (1 - dead)
        slack = shock * sstep(TSH + 0.5, 23.5, t) * 0.7 + dead
        u = {}
        def add(k, v):
            if v > 1e-4:
                u[k] = u.get(k, 0.0) + v
        for side in ("left", "right"):
            add(f"eyebrows-{side}-inner-up", 1.0 * fear + 0.9 * scream + 0.6 * pain + 0.5 * wide * shock)
            add(f"eyebrows-{side}-up", 0.6 * fear + 0.3 * wide * shock)
            add(f"eyebrows-{side}-down", 0.55 * scream * (1 - fear) + 0.75 * pain + 0.5 * brace)
            add(f"eye-{side}-opened-up", 1.0 * fear * (1 - brace) + 0.55 * wide * shock)
            add(f"eye-{side}-slit", 0.5 * scream * (1 - fear))
            add(f"eye-{side}-closure", 0.7 * brace + 0.65 * pain + (0.42 if side == "left" else 0.52) * slack)
            add(f"nose-{side}-elevation", 0.35 * scream + 0.6 * pain)
            add(f"nose-{side}-dilatation", 0.5 * fear + 0.3 * scream)
        add("mouth-retraction", 0.35 * fear + 0.6 * scream + 0.2 * pain)
        add("mouth-depression-retraction", 0.3 * scream + 0.7 * pain * (1 - scream))
        add("mouth-upward-retraction", 0.4 * pain * (1 - scream))
        add("neck-platysma", 0.35 * fear + 1.0 * scream + 0.6 * pain)
        add("mouth-depression", 0.3 * slack)
        # blinks (calm only; none once he is dead)
        blink = twitch(t, [1.4, 4.7, 7.9, 10.3, 11.65, 18.6, 20.3, 22.2], 0.05, 0.12)
        blink *= (1 - fear) * (1 - fight) * (1 - dead)
        for side in ("left", "right"):
            u[f"eye-{side}-closure"] = max(u.get(f"eye-{side}-closure", 0.0), blink)
        V.expr(pose, u)
        # jaw: driven by the voice, with a resting gape that depends on his state
        rest = 1.0 + 5 * fear + 3 * fight + 9 * shock * (1 - dead) + 10 * dead
        gain = {"scream": 25, "gasp": 22, "breath": 12, "grunt": 13, "wheeze": 12, "moan": 13}.get(kind, 0)
        jaw = rest + gain * env
        jaw += 6 * g.get("rip_jolt", 0.0) * (1 - dead) + 3 * g.get("bre", 0.0) * shock * (1 - dead) * 0.5
        V.face(pose, jaw=jaw)

    # ---------------- arms
    def _v_hand_spring(self, s, t, target, mode):
        """hand position with inertia: 'drive' (muscle, springy) or 'limp' (falls under gravity)."""
        st = getattr(self, "_vh", {})
        self._vh = st
        prev = st.get(s)
        if prev is None or abs(prev[2] - (t - 1.0 / FPS)) > 1e-4:
            st[s] = (target.copy(), Vector((0, 0, 0)), t)
            return target
        p, v, _ = prev
        p, v = p.copy(), v.copy()
        sub = 6
        dt = 1.0 / FPS / sub
        for _ in range(sub):
            if mode == "limp":
                k = 25.0
                acc = (target - p) * k
                acc.z = -9.81 * 0.85 if p.z > target.z + 0.002 else (target.z - p.z) * 400
                v = v * (1 - 4.0 * dt) + acc * dt
            else:
                k = 260.0 if mode == "drive" else 120.0
                acc = (target - p) * k - v * (2 * math.sqrt(k) * 0.9)
                v = v + acc * dt
            p = p + v * dt
            if p.z < 0.045:
                p.z = 0.045
                if v.z < 0:
                    v.z = -v.z * 0.15
                v.x *= 0.6
                v.y *= 0.6
        st[s] = (p, v, t)
        return p

    def victim_arms(self, t, vp, zp):
        V, Z, n = self.V, self.Z, self.nv
        V.rp.fk(vp)
        if t < self.T_IMPACT:
            walking, jolt, fear, back, brace, breath, trem = self._v_arm_state
            f, u, l = V.chest_frame(vp)
            for s, sg in (("L", 1), ("R", -1)):
                if s == "R":
                    base = V.hand_from_chest(vp, s, 0.27, -0.22, -0.10)                         # phone
                    startle = V.hand_from_chest(vp, s, 0.22, -0.09, -0.03)
                    guard = V.hand_from_chest(vp, s, 0.22, 0.03, -0.07)
                    cover = V.hand_from_chest(vp, s, 0.19, 0.19, -0.15)
                else:
                    ph = math.sin(t * 2 * math.pi * 0.9) * walking
                    base = V.hand_from_chest(vp, s, 0.06 + 0.13 * ph, -0.52, 0.05)
                    startle = V.hand_from_chest(vp, s, 0.20, -0.13, 0.08)
                    guard = V.hand_from_chest(vp, s, 0.27, 0.11, -0.02)
                    cover = V.hand_from_chest(vp, s, 0.24, 0.23, -0.09)
                tgt = base.lerp(startle, clamp(max(jolt, 0.75 * fear)) * (1 - back))
                tgt = tgt.lerp(guard, back).lerp(cover, brace)
                tgt += Vector((0.006 * n(t + 3 * sg, 9.0), 0.006 * n(t + 5 * sg, 8.0), 0.006 * n(t + 7 * sg, 10.0))) * fear
                # palm out toward the attacker, fingers up and splayed
                out = l * sg
                palm_out = V.hand_frame(s, (u * 0.85 + out * 0.15 + f * 0.1).normalized(), f)
                V.arm_ik_twist(vp, s, tgt, V.arm_pole(vp, s, back=0.5, out=0.9, down=0.7), palm_out, amount=0.85 * back)
                if s == "R":     # fingers wrapped round the phone, not a fist
                    V.fingers(vp, s, curl=lerp(0.42, 0.3, back), spread=0.15 + 0.4 * back, thumb=0.75)
                else:
                    V.fingers(vp, s, curl=lerp(0.25 + 0.2 * fear, 0.05, back), spread=0.3 * fear + 0.6 * back)
            V.eyes_at(vp, self._v_look if t >= self.T_V_NOTICE else self._phone_pt(vp), 1.0)
            return
        # ---------------- falling and on the ground
        T0, TS, TL = self.T_IMPACT, self.T_SEAT, self.T_LAND
        l0, cl, t0, t1, rip, ce = self.BITES[0]
        TSH = self.T_SHOCK
        g = self._v_ground
        fight, bst, bside, bph = g["fight"], g["bst"], g["bside"], g["bph"]
        dead, shock = g["dead"], g["shock"]
        vroot, vfront, vaxis, vlat = self.v_frame(vp)
        bite = V.rp.point(vp, "neck01", self.vc.bite_point)
        zr = Z.rp
        zh0 = zr.h_rest["head"]
        zface = zr.point(zp, "head", zh0 + Vector((0.0, -0.085, 0.07)))
        zbrow_r = zr.point(zp, "head", zh0 + Vector((-0.065, -0.06, 0.11)))      # zombie's right temple
        zjaw_l = zr.point(zp, "jaw", zr.h_rest["jaw"] + Vector((0.05, -0.04, -0.03)))
        zsh = {k: zp["H"]["upperarm01." + k] for k in ("L", "R")}
        zchest = zp["H"]["spine01"] + zp["D"]["spine01"] @ Vector((0, -0.10, 0))
        zcollar = zp["H"]["neck01"] + zp["D"]["neck01"] @ Vector((0, -0.07, -0.02))
        for s, sg in (("L", 1), ("R", -1)):
            sh = vp["H"]["upperarm01." + s]
            lead = 1.0 if bside == sg else 0.4
            if t < TL + 0.05:
                # clutching at the attacker's shoulders on the way down, the far arm thrown out
                # guard crushed between the bodies, then flung out to break the fall, hitting the asphalt
                w = clamp((t - T0) / (TL - T0))
                guard = V.hand_from_chest(vp, s, 0.20, 0.06, -0.04)
                fling = sh + vlat * (sg * 0.40) + vaxis * 0.24 + vfront * 0.22
                slap = sh + vlat * (sg * 0.46) + vaxis * 0.12
                slap.z = 0.05
                tgt = guard.lerp(fling, sstep(0.25, 0.7, w + (0.06 if s == "R" else 0.0)))
                tgt = tgt.lerp(slap, sstep(0.82, 1.0, w))
                mode, hand_rot, curl, spread = "drive", None, 0.3, 0.7
            else:
                # --- fighting: palm on the zombie's face / temple and on its shoulder or jaw, shoving
                if s == "L":
                    contact = zbrow_r.lerp(zface, 0.25 + 0.2 * n(t, 0.8))
                else:
                    contact = zsh["L"].lerp(zjaw_l, clamp(0.35 + 0.5 * bst * lead)).lerp(zcollar, 0.25 * (1 - bst))
                away = (sh - contact).normalized()
                slip = 0.05 * math.sin(math.pi * clamp((bph - 0.6) / 0.6)) * (1 if bph > 0.6 else 0)
                push = contact + away * (0.085 - 0.03 * bst * lead) + vaxis * slip
                push += Vector((0.008 * n(t + 5 * sg, 3.0), 0.008 * n(t + 9 * sg, 3.3), 0.006 * n(t + 2 * sg, 3.6))) * fight
                # --- shock: arms drop; his left hand finds the wound and presses it, then slips away
                lie = sh + vlat * (sg * (0.40 if s == "R" else 0.20)) + vaxis * (0.02 if s == "R" else 0.14)
                lie.z = 0.05
                wound = bite + vfront * 0.045 + vlat * 0.02 - vaxis * 0.02
                hold = sstep(17.7, 18.5, t) * (1 - sstep(20.5, 21.05, t)) if s == "L" else 0.0
                reach = 0.0
                if s == "R":     # feeble reach for the zombie's shoulder at the second bite
                    reach = sstep(21.72, 22.1, t) * (1 - sstep(22.35, 22.6, t))
                tgt = push
                active = clamp((fight - 0.12) * 1.6)
                if active < 1.0:
                    tgt = lie.lerp(push, active)
                if hold > 0:
                    tgt = tgt.lerp(wound, hold)
                if reach > 0:
                    tgt = tgt.lerp(zsh["L"] + vfront * 0.05, reach * 0.6)
                mode = "drive" if (active > 0.3 or hold > 0.05 or reach > 0.05) else "limp"
                f_dir = (contact - sh).normalized()
                palm = V.hand_frame(s, (f_dir + vaxis * 0.6).normalized(), f_dir)
                hand_rot = palm if active > 0.2 else None
                curl = lerp(0.42 + 0.1 * dead, 0.12 + 0.25 * (1 - bst), active)
                curl = lerp(curl, 0.55, hold) + 0.18 * g["agonal"] * (1 if s == "R" else 0.5)
                spread = lerp(0.12, 0.65, active)
            tgt = self._v_hand_spring(s, t, tgt, mode)
            pole = vlat * sg * 0.8 - vaxis * 0.5 - vfront * 0.25
            V.arm_ik_twist(vp, s, tgt, pole, hand_rot, amount=0.7)
            V.fingers(vp, s, curl=curl, spread=spread, thumb=0.45)
        # eyes: on the attacker while he fights, then staring past it, then fixed
        if t < TSH + 0.4:
            V.eyes_at(vp, zface, 1.0)
        else:
            V.eyes_at(vp, vp["H"]["head"] + vfront * 2.0 - vaxis * 0.4 + vlat * 0.3, 0.8)

    def _phone_pt(self, vp):
        return self.V.hand_from_chest(vp, "R", 0.27, -0.22, -0.10) + Vector((0, 0, 0.05))
