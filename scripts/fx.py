"""Gore / prop FX driven by the recorded per-frame animation data."""
import math
import random

import bpy
import bmesh
from mathutils import Vector, Quaternion, Matrix

import shading as sh
from anim import keys, sstep, clamp, FPS, NFRAMES


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


def flesh_material():
    mat = bpy.data.materials.new("Flesh")
    nb = sh.NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    n = nb.n("ShaderNodeTexNoise", in_Scale=90.0, in_Detail=8.0)
    nb.link(tc.outputs["Object"], n.inputs["Vector"])
    r = sh.ramp(nb, n.outputs["Fac"], [(0.35, (0.05, 0.002, 0.002)), (0.55, (0.32, 0.03, 0.03)),
                                      (0.7, (0.55, 0.22, 0.18)), (0.8, (0.6, 0.45, 0.35))])
    p, o = sh.principled(nb, Base_Color=r.outputs["Color"], Roughness=0.12)
    p.inputs["Subsurface Weight"].default_value = 0.4
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.15)
    p.inputs["Subsurface Scale"].default_value = 0.004
    p.inputs["Coat Weight"].default_value = 0.8
    return mat


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


def build(choreo, zombie, victim):
    ev = choreo.events["frames"]
    blood = sh.blood_material()
    fm = flesh_material()

    # --- wound / mouth blood spreading on skin and clothes
    animate_blood_radius(victim, [(15.2, 0.0), (15.5, 0.035), (17.4, 0.08), (19.0, 0.12), (22.0, 0.15),
                                  (25.2, 0.22), (30.0, 0.30)])
    animate_blood_radius(zombie, [(15.3, 0.0), (15.8, 0.02), (17.5, 0.04), (21.0, 0.055), (25.3, 0.075),
                                  (30.0, 0.09)])

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
        s = keys(T(f), [(15.6, 0.001), (18.0, 0.14), (22.0, 0.3), (26.0, 0.45), (30.0, 0.58)])
        _key_obj(pool, f, scale=(s, s, 1))
    pool.location.z = 0.004

    # --- torn flesh held in the teeth during chewing
    chunks = []
    for ci, (t_on, t_off) in enumerate([(17.38, 21.1), (25.15, 30.5)]):
        ch = make_blob(f"Chunk{ci}", 0.028, 10 + ci, fm, detail=3, rough=0.45)
        ch.scale = (1.0, 0.7, 1.5)
        chunks.append((ch, t_on, t_off))
    for ch, t_on, t_off in chunks:
        for f in range(1, NFRAMES + 1):
            t = T(f)
            e = ev[f]
            jawD = Quaternion(e["z_jaw_D"])
            mouth = Vector(e["z_mouth"])
            pos = mouth + jawD @ Vector((0.0, -0.012, -0.012))
            vis = 1.0 if t_on <= t <= t_off else 0.0
            # shrinks with chewing
            shrink = clamp(1 - 0.75 * (t - t_on) / max(0.1, (t_off - t_on))) if vis else 0
            s = 0.0001 + vis * shrink
            _key_obj(ch, f, loc=pos if vis else Vector((0, 0, -50)), rot=jawD, scale=(s, 0.75 * s, 1.4 * s))
        for fc in ch.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT" if fc.data_path == "scale" else "LINEAR"

    # --- blood drops: spurts during tearing + drips from the chin while chewing
    drops = []
    rnd = random.Random(42)
    spurts = []
    for (t0, t1, n) in ((15.25, 15.6, 6), (16.7, 17.5, 16), (22.0, 22.4, 6), (24.1, 25.25, 18)):
        for i in range(n):
            spurts.append(("spurt", rnd.uniform(t0, t1)))
    for (t0, t1, n) in ((17.6, 21.0, 14), (25.4, 30.0, 18)):
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
            p0 = Vector(e0["z_chin"])
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
    return dict(pool=pool, chunks=chunks, drops=drops, phone=phone)
