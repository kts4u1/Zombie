"""Gore / prop FX driven by the recorded per-frame animation data."""
import math
import os
import random

import bpy
import bmesh
from mathutils import Vector, Quaternion, Matrix

import shading as sh
from anim import keys, keys_lin, sstep, clamp, FPS, NFRAMES


def T(f):
    return (f - 1) / FPS


def _key_obj(ob, f, loc=None, rot=None, scale=None):
    if loc is not None:
        ob.location = loc
        ob.keyframe_insert("location", frame=f)
    if rot is not None:
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = rot
        ob.keyframe_insert("rotation_quaternion", frame=f)
    if scale is not None:
        ob.scale = scale
        ob.keyframe_insert("scale", frame=f)


def animate_blood_radius(char, pts):
    for nd in char.blood_radius_nodes():
        sock = nd.outputs[0]
        for f in range(1, NFRAMES + 1, 2):
            sock.default_value = keys(T(f), pts)
            sock.keyframe_insert("default_value", frame=f)


def make_blob(name, radius, seed, mat, detail=3, rough=0.35):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=detail, radius=radius, location=(0, 0, -50))
    ob = bpy.context.active_object
    ob.name = name
    rnd = random.Random(seed)
    for v in ob.data.vertices:
        k = 1 + rough * (rnd.random() - 0.5) * 2
        v.co *= k
    ob.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return ob


# ----------------------------------------------------------------------------
# torn flesh strip: clamped between the incisors, peeled off the wound, hanging
# under gravity, drawn into the mouth chew by chew (verlet rope -> PC2 cache)
# ----------------------------------------------------------------------------
CACHE_DIR = "/tmp"
STRIP_N = 12          # nodes: 0 inside the mouth, 1 at the incisors, 2.. free
STRIP_K = 10          # vertices per cross-section ring


def skin_flap_material():
    mat = bpy.data.materials.new("FlapSkin")
    nb = sh.NB(mat)
    uv = nb.n("ShaderNodeUVMap")
    n = nb.n("ShaderNodeTexNoise", in_Scale=14.0, in_Detail=6.0)
    nb.link(uv.outputs["UV"], n.inputs["Vector"])
    r = sh.ramp(nb, n.outputs["Fac"], [(0.36, (0.09, 0.005, 0.004)), (0.5, (0.3, 0.07, 0.055)),
                                      (0.66, (0.5, 0.31, 0.25))])
    p, o = sh.principled(nb, Base_Color=r.outputs["Color"], Roughness=0.3)
    p.inputs["Subsurface Weight"].default_value = 0.35
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.25)
    p.inputs["Subsurface Scale"].default_value = 0.006
    p.inputs["Coat Weight"].default_value = 0.6
    p.inputs["Coat Roughness"].default_value = 0.08
    return mat


def raw_flesh_material():
    mat = bpy.data.materials.new("RawFlesh")
    nb = sh.NB(mat)
    uv = nb.n("ShaderNodeUVMap")
    mp = nb.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 9.0, 1.0)
    nb.link(uv.outputs["UV"], mp.inputs["Vector"])
    w = nb.n("ShaderNodeTexWave", in_Scale=6.0, in_Distortion=7.0, in_Detail=4.0)
    nb.link(mp.outputs["Vector"], w.inputs["Vector"])
    n = nb.n("ShaderNodeTexNoise", in_Scale=40.0, in_Detail=8.0)
    nb.link(uv.outputs["UV"], n.inputs["Vector"])
    mix = nb.n("ShaderNodeMath", operation="MULTIPLY")
    nb.link(w.outputs["Fac"], mix.inputs[0]); nb.link(n.outputs["Fac"], mix.inputs[1])
    r = sh.ramp(nb, mix.outputs[0], [(0.05, (0.035, 0.0015, 0.0015)), (0.2, (0.16, 0.007, 0.006)),
                                    (0.38, (0.34, 0.03, 0.025)), (0.6, (0.42, 0.11, 0.08))])
    p, o = sh.principled(nb, Base_Color=r.outputs["Color"], Roughness=0.18)
    p.inputs["Subsurface Weight"].default_value = 0.45
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.15)
    p.inputs["Subsurface Scale"].default_value = 0.004
    p.inputs["Coat Weight"].default_value = 1.0
    p.inputs["Coat Roughness"].default_value = 0.04
    bump = nb.n("ShaderNodeBump", in_Strength=0.5, in_Distance=0.002)
    nb.link(mix.outputs[0], bump.inputs["Height"])
    nb.link(bump.outputs["Normal"], p.inputs["Normal"])
    return mat


def _strip_profile(K, seed, n_rings):
    """per ring half width / half thickness / bulge: a ragged strip of skin that
    widens into a lump of torn muscle at its free end."""
    rnd = random.Random(seed)
    prof = []
    for i in range(n_rings):
        s = i / (n_rings - 1)
        w = keys_lin(s, [(0.0, 0.015), (0.35, 0.018), (0.7, 0.024), (0.88, 0.027), (1.0, 0.017)])
        h = keys_lin(s, [(0.0, 0.0045), (0.35, 0.0055), (0.7, 0.011), (0.88, 0.015), (1.0, 0.009)])
        w *= 1 + 0.32 * (rnd.random() - 0.5)
        h *= 1 + 0.3 * (rnd.random() - 0.5)
        bulge = -0.45 * h * sstep(0.45, 0.85, s)          # meat bulges on the raw side
        jit = [1 + 0.36 * (rnd.random() - 0.5) for _ in range(K)]
        prof.append((w, h, jit, bulge))
    return prof


def _ring(K, w, h, jit, bulge=0.0):
    pts = []
    for k in range(K):
        a = 2 * math.pi * k / K
        c, s = math.cos(a), math.sin(a)
        x = w * math.copysign(abs(c) ** 0.7, c) * jit[k]
        y = h * math.copysign(abs(s) ** 0.75, s) * (0.75 + 0.5 * jit[k]) + bulge * (1 if s < 0 else 0.3)
        pts.append((x, y))
    return pts


def _make_strip_mesh(name, seed, skin_mat, flesh_mat):
    N, K = STRIP_N, STRIP_K
    prof = _strip_profile(K, seed, N)
    verts, faces, mats, uvs = [], [], [], []
    for i in range(N):
        for (x, y) in _ring(K, *prof[i]):
            verts.append((x, y, -0.01 * i))
    c0 = len(verts); verts.append((0, 0, 0.004))
    c1 = len(verts); verts.append((0, 0, -0.01 * (N - 1) - 0.004))
    for i in range(N - 1):
        for k in range(K):
            a, b = i * K + k, i * K + (k + 1) % K
            faces.append((a, b, b + K, a + K))
            am = 2 * math.pi * (k + 0.5) / K
            mats.append(0 if math.sin(am) > 0.45 else 1)
            uvs.append([(k / K, i / N), ((k + 1) / K, i / N), ((k + 1) / K, (i + 1) / N), (k / K, (i + 1) / N)])
    for k in range(K):
        faces.append((c0, (k + 1) % K, k)); mats.append(1)
        uvs.append([(0.5, 0), ((k + 1) / K, 0.02), (k / K, 0.02)])
        o = (N - 1) * K
        faces.append((c1, o + k, o + (k + 1) % K)); mats.append(1)
        uvs.append([(0.5, 1), (k / K, 0.98), ((k + 1) / K, 0.98)])
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.materials.append(skin_mat)
    me.materials.append(flesh_mat)
    uvl = me.uv_layers.new(name="UVMap")
    li = 0
    for pi, poly in enumerate(me.polygons):
        poly.material_index = mats[pi]
        poly.use_smooth = True
        for j, _ in enumerate(poly.loop_indices):
            uvl.data[li].uv = uvs[pi][j]
            li += 1
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob, prof


def _lerpv(a, b, w):
    return Vector(a).lerp(Vector(b), w)


def simulate_strip(choreo, ev, bi, length, seed):
    """returns per-frame node positions (list of Vectors or None) and tip positions."""
    l0, cl, t0, t1, rip, ce = choreo.BITES[bi]
    N = STRIP_N
    g = Vector((0, 0, -9.81))
    sub = 12
    dt = 1.0 / (FPS * sub)
    f_start = max(1, int((t1 - 0.25) * FPS))
    # chewing schedule: drawn into the mouth in steps, one per chew (1.55 Hz)
    c_start, c_end = rip + 0.5, min(ce - 0.3, 31.0)
    n_chew = max(1.0, (c_end - c_start) * 1.55)

    def eaten(t):
        if t < c_start:
            return 0.0
        x = (t - c_start) * 1.55
        i = math.floor(x)
        ph = x - i
        return clamp((i + sstep(0.05, 0.45, ph)) / n_chew)

    def inputs(fr):
        """frame (float) -> anchors."""
        fr = max(1.0, min(float(NFRAMES), fr))
        fa = int(math.floor(fr)); w = fr - fa
        fa = max(1, min(NFRAMES, fa)); fb = max(1, min(NFRAMES, fa + 1))
        A, B = ev[fa], ev[fb]
        tup = _lerpv(A["z_tup"], B["z_tup"], w)
        tlo = _lerpv(A["z_tlo"], B["z_tlo"], w)
        Dh = Quaternion(A["z_head_D"]).slerp(Quaternion(B["z_head_D"]), w)
        Dj = Quaternion(A["z_jaw_D"]).slerp(Quaternion(B["z_jaw_D"]), w)
        wound = _lerpv(A["v_bite"], B["v_bite"], w)
        chin = _lerpv(A["z_chin"], B["z_chin"], w)
        neck = _lerpv(A["z_neck"], B["z_neck"], w)
        fwd = Dh @ Vector((0, -1, 0))
        mid = (tup + tlo) * 0.5
        return dict(n0=mid - fwd * 0.026, n1=mid + fwd * 0.003, fwd=fwd, lat=Dh @ Vector((1, 0, 0)),
                    jfwd=Dj @ Vector((0, -1, 0)), wound=wound, chin=chin, neck=neck,
                    gap=(tup - tlo).length)

    out, tips, info = {}, {}, {}
    pos = prev = None
    attached = True
    visible = False
    for f in range(f_start, NFRAMES + 1):
        for s in range(sub):
            fr = f - 1 + (s + 1) / sub      # sim time in frames, ends exactly on frame f
            t = (fr - 1) / FPS
            I = inputs(fr)
            dist = (I["n1"] - I["wound"]).length
            if attached:
                Lf = clamp(dist * 1.06, 0.004, length)
                if dist > length * 1.12 or t > rip + 0.18:
                    attached = False
            else:
                Lf = length * (1.0 - eaten(t))
            seg = max(Lf, 1e-4) / (N - 2)
            if pos is None:
                # lay the strip from the teeth to the wound
                pos = [I["n0"], I["n1"]] + [I["n1"].lerp(I["wound"], (i - 1) / (N - 2)) for i in range(2, N)]
                prev = [p.copy() for p in pos]
            # integrate
            last_free = N - 2 if attached else N - 1
            for i in range(2, N):
                if i > last_free:
                    continue
                vel = (pos[i] - prev[i]) * 0.993
                prev[i] = pos[i].copy()
                pos[i] = pos[i] + vel + g * (dt * dt)
            prev[0], prev[1] = pos[0], pos[1]
            pos[0], pos[1] = I["n0"].copy(), I["n1"].copy()
            if attached:
                prev[N - 1] = pos[N - 1]
                pos[N - 1] = I["wound"].copy()
            # teeth hold the root of the strip: it leaves the mouth pointing forward/down
            hold = I["n1"] + (I["fwd"] * 0.75 + Vector((0, 0, -0.65))).normalized() * seg
            for it in range(10):
                pos[2] = pos[2].lerp(hold, 0.05 if attached else 0.18)
                for i in range(1, N - 1):
                    a, b = pos[i], pos[i + 1]
                    d = b - a
                    L = d.length
                    if L < 1e-9:
                        continue
                    corr = d * ((L - seg) / L)
                    wa = 0.0 if i <= 1 else 0.5
                    wb = 0.0 if (attached and i + 1 == N - 1) else 0.5
                    if wa + wb == 0:
                        continue
                    k = 1.0 / (wa + wb)
                    pos[i] = a + corr * (wa * k)
                    pos[i + 1] = b - corr * (wb * k)
                # bending stiffness: keep i, i+2 apart
                for i in range(1, N - 2):
                    a, b = pos[i], pos[i + 2]
                    d = b - a
                    L = d.length
                    if 1e-9 < L < seg * 1.7:
                        corr = d * ((L - seg * 1.7) / L) * 0.25
                        if i >= 2:
                            pos[i] = a + corr * 0.5
                        if not (attached and i + 2 == N - 1):
                            pos[i + 2] = b - corr * 0.5
                # collisions: ground, the zombie's chin and neck
                for i in range(2, N):
                    p = pos[i]
                    if p.z < 0.008:
                        p.z = 0.008
                    for c, r in ((I["chin"] - I["jfwd"] * 0.032 + Vector((0, 0, 0.008)), 0.036),
                                 (I["neck"], 0.062)):
                        d = p - c
                        if d.length < r:
                            pos[i] = c + d.normalized() * r
        if not visible and (not attached or dist > 0.03):
            visible = True
        if visible and Lf > 0.006:
            out[f] = [p.copy() for p in pos]
            info[f] = I
            tips[f] = pos[N - 1].copy()
        else:
            out[f] = None
    return out, tips, info


def build_strip(choreo, ev, bi, length, seed, skin_mat, flesh_mat):
    import struct
    import numpy as np
    ob, prof = _make_strip_mesh(f"FleshStrip{bi}", seed, skin_mat, flesh_mat)
    sim, tips, info = simulate_strip(choreo, ev, bi, length, seed)
    N, K = STRIP_N, STRIP_K
    nv = N * K + 2
    arr = np.zeros((NFRAMES, nv, 3), np.float32)
    arr[:, :, 2] = -50.0
    rings = [_ring(K, *prof[i]) for i in range(N)]
    for f in range(1, NFRAMES + 1):
        P = sim.get(f)
        if not P:
            continue
        I = info[f]
        Wd = None
        frames = []
        for i in range(N):
            if i == 0:
                T = (P[1] - P[0])
            elif i == N - 1:
                T = (P[i] - P[i - 1])
            else:
                T = (P[i + 1] - P[i - 1])
            T = T.normalized() if T.length > 1e-9 else Vector((0, 0, -1))
            if Wd is None:
                Wd = I["lat"] - T * I["lat"].dot(T)
            else:
                Wd = Wd - T * Wd.dot(T)
            Wd = Wd.normalized() if Wd.length > 1e-9 else T.orthogonal().normalized()
            Nn = T.cross(Wd)
            frames.append((T, Wd, Nn))
        for i in range(N):
            T, Wd, Nn = frames[i]
            for k, (x, y) in enumerate(rings[i]):
                arr[f - 1, i * K + k] = tuple(P[i] + Wd * x + Nn * y)
        arr[f - 1, N * K] = tuple(P[0] - frames[0][0] * 0.004)
        arr[f - 1, N * K + 1] = tuple(P[N - 1] + frames[N - 1][0] * 0.004)
    path = os.path.join(CACHE_DIR, f"flesh_strip{bi}.pc2")
    with open(path, "wb") as fh:
        fh.write(b"POINTCACHE2\0")
        fh.write(struct.pack("<iiffi", 1, nv, 1.0, 1.0, NFRAMES))
        fh.write(arr.tobytes())
    m = ob.modifiers.new("cache", "MESH_CACHE")
    m.cache_format = "PC2"
    m.filepath = path
    m.frame_start = 1.0
    m.interpolation = "LINEAR"
    # hide the object outside its life (cheaper BVH, no stray geometry)
    live = [f for f in range(1, NFRAMES + 1) if sim.get(f)]
    for f in (1, live[0] - 1, live[0], live[-1], live[-1] + 1):
        if 1 <= f <= NFRAMES:
            ob.hide_render = not (live[0] <= f <= live[-1])
            ob.keyframe_insert("hide_render", frame=f)
    return ob, tips, (live[0], live[-1])


def build(choreo, zombie, victim):
    ev = choreo.events["frames"]
    blood = sh.blood_material()

    # --- wound / mouth blood spreading on skin and clothes
    animate_blood_radius(victim, [(14.28, 0.0), (14.6, 0.035), (16.9, 0.08), (19.0, 0.12), (21.6, 0.15),
                                  (24.4, 0.22), (30.0, 0.30)])
    animate_blood_radius(zombie, [(14.3, 0.0), (14.8, 0.015), (17.0, 0.03), (21.0, 0.038), (24.5, 0.046),
                                  (30.0, 0.052)])

    # --- blood pool under the victim's neck (irregular, grows)
    bite0 = Vector(ev[400]["v_bite"])
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=1.0, fill_type="NGON", location=(bite0.x + 0.08, bite0.y, 0.004))
    pool = bpy.context.active_object
    pool.name = "BloodPool"
    rnd = random.Random(5)
    ph = [rnd.random() * 6.28 for _ in range(6)]
    for v in pool.data.vertices:
        a = math.atan2(v.co.y, v.co.x)
        r = v.co.length
        k = 1 + 0.18 * math.sin(3 * a + ph[0]) + 0.12 * math.sin(5 * a + ph[1]) + 0.07 * math.sin(11 * a + ph[2])
        v.co.x *= k * 1.0
        v.co.y *= k * 1.35
    pool.data.materials.append(blood)
    for f in range(1, NFRAMES + 1, 3):
        s = keys(T(f), [(14.8, 0.001), (17.0, 0.14), (21.0, 0.3), (25.0, 0.45), (30.0, 0.58)])
        _key_obj(pool, f, scale=(s, s, 1))
    pool.location.z = 0.004

    # --- torn flesh strips: gripped in the teeth, hanging, eaten chew by chew
    skin_m, flesh_m = skin_flap_material(), raw_flesh_material()
    strips, tips_all = [], {}
    for bi, (ln, seed) in enumerate(((0.09, 3), (0.10, 8))):
        ob, tips, life = build_strip(choreo, ev, bi, ln, seed, skin_m, flesh_m)
        strips.append(ob)
        tips_all.update(tips)
        print("strip", bi, "frames", life)

    # --- blood drops: spurts during tearing + drips from the chin while chewing
    drops = []
    rnd = random.Random(42)
    spurts = []
    for (t0, t1, n) in ((14.3, 14.6, 6), (15.55, 16.95, 16), (21.6, 21.9, 6), (22.6, 24.4, 18)):
        for i in range(n):
            spurts.append(("spurt", rnd.uniform(t0, t1)))
    for (t0, t1, n) in ((17.1, 21.0, 14), (24.6, 30.0, 18)):
        for i in range(n):
            spurts.append(("drip", rnd.uniform(t0, t1)))
    g = 9.81
    for i, (kind, ts) in enumerate(spurts):
        d = make_blob(f"Drop{i}", rnd.uniform(0.004, 0.009), 100 + i, blood, detail=2, rough=0.25)
        f0 = int(ts * FPS) + 1
        e0 = ev[min(NFRAMES, f0)]
        if kind == "spurt":
            p0 = Vector(e0["v_bite"]) + Vector((rnd.uniform(-0.03, 0.03), rnd.uniform(-0.03, 0.03), 0.04))
            v0 = Vector((rnd.uniform(-0.8, 1.2), rnd.uniform(-0.9, 0.9), rnd.uniform(0.8, 2.2)))
        else:
            # drips run off the hanging flesh (or the chin once it is eaten)
            p0 = tips_all[f0].copy() if f0 in tips_all else Vector(e0["z_chin"])
            v0 = Vector((rnd.uniform(-0.05, 0.05), rnd.uniform(-0.05, 0.05), -0.05))
        floor_z = 0.006 if kind == "spurt" else max(0.08, p0.z - 0.35)
        landed = None
        for f in range(max(1, f0 - 1), min(NFRAMES, f0 + 40) + 1):
            t = T(f) - ts
            if t < 0:
                _key_obj(d, f, loc=Vector((0, 0, -50)), scale=(1, 1, 1))
                continue
            pos = p0 + v0 * t + Vector((0, 0, -0.5 * g * t * t))
            vel = v0 + Vector((0, 0, -g * t))
            if pos.z < floor_z or landed:
                if landed is None:
                    landed = pos.copy(); landed.z = floor_z
                _key_obj(d, f, loc=landed, scale=(1.8, 1.8, 0.25))
                continue
            stretch = 1 + min(2.5, vel.length * 0.35)
            q = Vector((0, 0, 1)).rotation_difference(vel.normalized()) if vel.length > 1e-3 else Quaternion()
            _key_obj(d, f, loc=pos, rot=q, scale=(1, 1, stretch))
        drops.append(d)

    # --- phone in the victim's right hand (screen lights his face), dropped at impact
    pm = bpy.data.materials.new("PhoneBody")
    nb = sh.NB(pm)
    sh.principled(nb, Base_Color=(0.02, 0.02, 0.022, 1), Roughness=0.2, Metallic=0.6)
    scr = bpy.data.materials.new("PhoneScreen")
    nb = sh.NB(scr)
    e = nb.n("ShaderNodeEmission", in_Color=(0.75, 0.85, 1.0, 1), in_Strength=4.0)
    o = nb.n("ShaderNodeOutputMaterial")
    nb.link(e.outputs[0], o.inputs["Surface"])
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -50))
    phone = bpy.context.active_object
    phone.name = "Phone"
    phone.scale = (0.072, 0.0085, 0.15)
    bpy.ops.object.transform_apply(scale=True)
    phone.data.materials.append(pm)
    phone.data.materials.append(scr)
    # screen = face toward -Y local
    for poly in phone.data.polygons:
        if poly.normal.y < -0.9:
            poly.material_index = 1
    # rest-space hand frame
    rp = choreo.V.rp
    w0 = rp.h_rest["wrist.R"]
    k0 = rp.h_rest["finger3-1.R"]
    palm = (w0 + k0) * 0.5
    T_IMP = choreo.T_IMPACT
    drop_pos, drop_vel = None, None
    prev = None
    landed_at = None
    for f in range(1, NFRAMES + 1):
        t = T(f)
        e = ev[f]
        D = Quaternion(e["v_wristR_D"])
        H = Vector(e["v_wristR"])
        if t <= T_IMP:
            pos = H + D @ (palm - w0) + D @ Vector((0.02, -0.03, 0.0))
            # phone orientation: long axis along the hand, screen toward the face
            hand_dir = (D @ (k0 - w0)).normalized()
            up_hint = Vector((0, 0, 1))
            from rigpose import frame_rot
            rot = frame_rot(Vector((0, 0, 1)), Vector((0, -1, 0)), hand_dir, -(D @ Vector((0, 1, 0))))
            if prev is not None:
                drop_vel = (pos - prev) * FPS
            prev = pos
            drop_pos = pos
            drop_rot = rot
            _key_obj(phone, f, loc=pos, rot=rot)
        else:
            dt = t - T_IMP
            v0 = (drop_vel or Vector((0, 0, 0))) * 0.4 + Vector((0.3, -0.8, 0.6))
            pos = drop_pos + v0 * dt + Vector((0, 0, -0.5 * 9.81 * dt * dt))
            if pos.z < 0.01 or landed_at is not None:
                if landed_at is None:
                    landed_at = pos.copy()
                    landed_at.z = 0.006
                pos = landed_at
                rot = Quaternion((1, 0, 0), math.radians(90)) @ Quaternion((0, 0, 1), math.radians(30))
            else:
                rot = drop_rot @ Quaternion((1, 0.3, 0.2), dt * 14)
            _key_obj(phone, f, loc=pos, rot=rot)
    # landed phone: freeze after it hits the ground
    return dict(pool=pool, strips=strips, drops=drops, phone=phone)
