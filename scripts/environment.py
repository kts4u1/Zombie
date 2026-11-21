"""Night street environment: wet asphalt, sidewalks, buildings, sodium lamps."""
import math
import os
import random

import bpy
import bmesh
from mathutils import Vector

import shading as sh
from shading import NB, mix, math as smath, ramp, tex_node, principled


def _mat_textured(name, ph, base, scale=1.0, wet=0.0, tint=(1, 1, 1, 1), disp=None):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    mp = nb.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    nb.link(tc.outputs["Object"], mp.inputs["Vector"])
    diff = tex_node(nb, os.path.join(ph, f"{base}_diff_2k.jpg"), uv=mp.outputs["Vector"])
    rough = tex_node(nb, os.path.join(ph, f"{base}_rough_2k.jpg"), "Non-Color", uv=mp.outputs["Vector"])
    nor = tex_node(nb, os.path.join(ph, f"{base}_nor_gl_2k.jpg"), "Non-Color", uv=mp.outputs["Vector"])
    for t in (diff, rough, nor):
        t.projection = "BOX"
        t.projection_blend = 0.2
    col = mix(nb, 1.0, diff.outputs["Color"], tint, "MULTIPLY")
    r = rough.outputs["Color"]
    if wet > 0:
        # puddles: large-scale noise -> near-mirror, darker
        pn = nb.n("ShaderNodeTexNoise", in_Scale=0.35, in_Detail=6.0, in_Roughness=0.6)
        nb.link(tc.outputs["Object"], pn.inputs["Vector"])
        pm = ramp(nb, pn.outputs["Fac"], [(0.52, (0, 0, 0)), (0.6, (1, 1, 1))]).outputs["Color"]
        damp = smath(nb, "MULTIPLY", wet, 0.6)
        col = mix(nb, smath(nb, "ADD", smath(nb, "MULTIPLY", pm, 0.5), damp), col, (0.0, 0.0, 0.0, 1), "MIX")
        rw = smath(nb, "MULTIPLY", r, smath(nb, "SUBTRACT", 1.0, smath(nb, "MULTIPLY", wet, 0.5)))
        r = mix(nb, pm, rw, (0.02, 0.02, 0.02, 1))
    p, out = principled(nb, Base_Color=col, Roughness=r)
    nm = nb.n("ShaderNodeNormalMap", in_Strength=1.0)
    nb.link(nor.outputs["Color"], nm.inputs["Color"])
    nb.link(nm.outputs["Normal"], p.inputs["Normal"])
    return mat


def _box(name, loc, size, mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        ob.data.materials.append(mat)
    return ob


def building_material(ph):
    mat = bpy.data.materials.new("Building")
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    obj = tc.outputs["Object"]
    mp = nb.n("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.5, 0.5, 0.5)
    nb.link(obj, mp.inputs["Vector"])
    diff = tex_node(nb, os.path.join(ph, "brick_wall_001_diff_2k.jpg"), uv=mp.outputs["Vector"])
    nor = tex_node(nb, os.path.join(ph, "brick_wall_001_nor_gl_2k.jpg"), "Non-Color", uv=mp.outputs["Vector"])
    for t in (diff, nor):
        t.projection = "BOX"
        t.projection_blend = 0.2
    col = mix(nb, 1.0, diff.outputs["Color"], (0.55, 0.5, 0.48, 1), "MULTIPLY")
    # window grid (world space): cells 3.2m wide x 3.3m tall
    geo = nb.n("ShaderNodeNewGeometry")
    sep = nb.n("ShaderNodeSeparateXYZ")
    nb.link(geo.outputs["Position"], sep.inputs[0])
    u = smath(nb, "FRACT", smath(nb, "DIVIDE", sep.outputs["Y"], 3.2))
    v = smath(nb, "FRACT", smath(nb, "DIVIDE", sep.outputs["Z"], 3.3))
    wu = smath(nb, "MULTIPLY", smath(nb, "GREATER_THAN", u, 0.25), smath(nb, "LESS_THAN", u, 0.75))
    wv = smath(nb, "MULTIPLY", smath(nb, "GREATER_THAN", v, 0.3), smath(nb, "LESS_THAN", v, 0.82))
    above = smath(nb, "GREATER_THAN", sep.outputs["Z"], 3.0)
    win = smath(nb, "MULTIPLY", smath(nb, "MULTIPLY", wu, wv), above)
    # per-cell random: lit or not
    cy = smath(nb, "FLOOR", smath(nb, "DIVIDE", sep.outputs["Y"], 3.2))
    cz = smath(nb, "FLOOR", smath(nb, "DIVIDE", sep.outputs["Z"], 3.3))
    cv = nb.n("ShaderNodeCombineXYZ")
    nb.link(cy, cv.inputs[0]); nb.link(cz, cv.inputs[1]); nb.link(sep.outputs["X"], cv.inputs[2])
    wn = nb.n("ShaderNodeTexWhiteNoise")
    wn.noise_dimensions = "3D"
    nb.link(cv.outputs[0], wn.inputs["Vector"])
    lit = smath(nb, "GREATER_THAN", wn.outputs["Value"], 0.86)
    glass = (0.01, 0.012, 0.015, 1)
    col = mix(nb, win, col, glass)
    rough = mix(nb, win, (0.85, 0.85, 0.85, 1), (0.05, 0.05, 0.05, 1))
    em_col = ramp(nb, wn.outputs["Color"], [(0.0, (1.0, 0.62, 0.32)), (1.0, (0.85, 0.78, 0.6))])
    em_str = smath(nb, "MULTIPLY", smath(nb, "MULTIPLY", win, lit), 0.7)
    p, out = principled(nb, Base_Color=col, Roughness=rough)
    nb.link(em_col.outputs["Color"], p.inputs["Emission Color"])
    nb.link(em_str, p.inputs["Emission Strength"])
    nm = nb.n("ShaderNodeNormalMap", in_Strength=0.8)
    nb.link(nor.outputs["Color"], nm.inputs["Color"])
    nb.link(nm.outputs["Normal"], p.inputs["Normal"])
    return mat


def metal_material(name, col=(0.05, 0.05, 0.05, 1), rough=0.5):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    p, out = principled(nb, Base_Color=col, Roughness=rough, Metallic=0.8)
    return mat


def emissive_material(name, col, strength):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    e = nb.n("ShaderNodeEmission")
    e.inputs["Color"].default_value = col
    e.inputs["Strength"].default_value = strength
    out = nb.n("ShaderNodeOutputMaterial")
    nb.link(e.outputs[0], out.inputs["Surface"])
    return mat


def street_lamp(name, x, y, side, metal, flicker=None, energy=2200):
    """pole at (x,y), arm reaching over the road (toward x=0)."""
    pole = _box(name + "_pole", (x, y, 3.2), (0.14, 0.14, 6.4), metal)
    arm_len = 1.8
    ax = x - side * arm_len / 2
    arm = _box(name + "_arm", (ax, y, 6.35), (arm_len, 0.1, 0.1), metal)
    hx = x - side * arm_len
    head = _box(name + "_head", (hx, y, 6.28), (0.6, 0.28, 0.14), metal)
    bulb = _box(name + "_bulb", (hx, y, 6.19), (0.48, 0.2, 0.03),
                emissive_material(name + "_glow", (1.0, 0.55, 0.18, 1), 60.0))
    bulb.visible_shadow = False
    ld = bpy.data.lights.new(name + "_light", "SPOT")
    ld.energy = energy
    ld.color = (1.0, 0.58, 0.24)
    ld.spot_size = math.radians(125)
    ld.spot_blend = 0.75
    ld.shadow_soft_size = 0.25
    lo = bpy.data.objects.new(name + "_light", ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = (hx, y, 6.1)
    lo.rotation_euler = (0, 0, 0)
    if flicker:
        for f, e in flicker:
            ld.energy = energy * e
            ld.keyframe_insert("energy", frame=f)
            for kp in ld.animation_data.action.fcurves[0].keyframe_points:
                kp.interpolation = "CONSTANT"
        # bulb glow follows
        nd = bulb.data.materials[0].node_tree.nodes["Emission"]
        for f, e in flicker:
            nd.inputs["Strength"].default_value = 60 * e
            nd.inputs["Strength"].keyframe_insert("default_value", frame=f)
    return lo


def build(ph, seed=3):
    random.seed(seed)
    sc = bpy.context.scene
    # --- world: dim night sky HDRI
    world = bpy.data.worlds.new("Night")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    hdr = os.path.join(ph, "moonless_golf_2k.hdr")
    if os.path.exists(hdr):
        env.image = bpy.data.images.load(hdr)
    tco = nt.nodes.new("ShaderNodeTexCoord")
    mpn = nt.nodes.new("ShaderNodeMapping")
    mpn.inputs["Rotation"].default_value = (0, 0, 2.2)
    nt.links.new(tco.outputs["Generated"], mpn.inputs["Vector"])
    nt.links.new(mpn.outputs["Vector"], env.inputs["Vector"])
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.6
    out = nt.nodes.new("ShaderNodeOutputWorld")
    tint = nt.nodes.new("ShaderNodeMix")
    tint.data_type = "RGBA"
    tint.blend_type = "MULTIPLY"
    tint.inputs[0].default_value = 1.0
    nt.links.new(env.outputs["Color"], tint.inputs[6])
    tint.inputs[7].default_value = (0.45, 0.52, 0.85, 1)
    nt.links.new(tint.outputs[2], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    world.mist_settings.start = 6.0
    world.mist_settings.depth = 110.0
    world.mist_settings.falloff = "QUADRATIC"

    asphalt = _mat_textured("Asphalt", ph, "asphalt_02", scale=0.33, wet=0.55, tint=(0.62, 0.62, 0.64, 1))
    pave = _mat_textured("Pavement", ph, "concrete_pavement", scale=0.4, wet=0.25, tint=(0.7, 0.7, 0.7, 1))
    metal = metal_material("LampMetal")

    y0, y1 = 25.0, -110.0
    L = y0 - y1
    yc = (y0 + y1) / 2
    road = _box("Road", (0, yc, -0.05), (8.0, L, 0.1), asphalt)
    for s in (-1, 1):
        _box(f"Sidewalk{s}", (s * 5.75, yc, 0.075), (3.5, L, 0.15), pave)
        _box(f"Curb{s}", (s * 4.06, yc, 0.07), (0.14, L, 0.16), pave)
    # lane markings (worn dashes)
    paint = bpy.data.materials.new("Paint")
    nb = NB(paint)
    tc = nb.n("ShaderNodeTexCoord")
    nn = nb.n("ShaderNodeTexNoise", in_Scale=8.0, in_Detail=6.0)
    nb.link(tc.outputs["Object"], nn.inputs["Vector"])
    a = ramp(nb, nn.outputs["Fac"], [(0.35, (0, 0, 0)), (0.5, (1, 1, 1))])
    p, o = principled(nb, Base_Color=(0.55, 0.52, 0.42, 1), Roughness=0.5, Alpha=a.outputs["Color"])
    y = y0
    while y > y1:
        _box("Dash", (0, y, 0.002), (0.12, 3.0, 0.004), paint)
        y -= 7.0

    # buildings both sides with gaps (alleys)
    bmat = building_material(ph)
    for s in (-1, 1):
        y = y0
        while y > y1:
            w = random.uniform(12, 24)
            h = random.uniform(9, 24)
            depth = 10
            _box(f"Bldg{s}", (s * (7.5 + depth / 2), y - w / 2, h / 2), (depth, w - 0.4, h), bmat)
            y -= w + random.choice([0.0, 0.0, 2.5])

    # street lamps (alternating sides), one flickering near the zombie
    lamps = []
    flick = []
    rng = random.Random(7)
    f = 1
    on = True
    while f < 760:
        dur = rng.choice([2, 3, 4, 40, 60, 90]) if on else rng.choice([1, 2, 3, 5])
        flick.append((f, 1.0 if on else rng.choice([0.0, 0.08, 0.25])))
        f += dur
        on = not on
    lamp_ys = [(4.0, -1), (-5.5, 1), (-20.0, -1), (-38.0, 1), (-55.0, -1), (-62.5, 1), (-71.5, -1), (-88.0, 1)]
    for i, (ly, side) in enumerate(lamp_ys):
        x = side * 4.5
        lamps.append(street_lamp(f"Lamp{i}", x, ly, side, metal, flicker=flick if i == 0 else None))

    # moonlight
    md = bpy.data.lights.new("Moon", "SUN")
    md.energy = 0.22
    md.color = (0.55, 0.65, 1.0)
    md.angle = math.radians(1.0)
    mo = bpy.data.objects.new("Moon", md)
    sc.collection.objects.link(mo)
    mo.rotation_euler = (math.radians(55), 0, math.radians(-140))

    # clutter: trash bags and a dumpster
    bag = bpy.data.materials.new("TrashBag")
    nb = NB(bag)
    p, o = principled(nb, Base_Color=(0.01, 0.01, 0.012, 1), Roughness=0.25)
    for (x, y) in [(5.0, 2.5), (5.3, 1.6), (4.8, 0.7), (-4.9, -58.8), (-5.2, -59.6), (5.1, -66.0), (-5.0, -22.0)]:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.38, location=(x, y, 0.4))
        ob = bpy.context.active_object
        ob.scale = (1.0, rng.uniform(0.8, 1.1), rng.uniform(0.7, 0.95))
        d = ob.modifiers.new("d", "DISPLACE")
        t = bpy.data.textures.new("bagtex", "CLOUDS")
        t.noise_scale = 0.25
        d.texture = t
        d.strength = 0.12
        ob.data.materials.append(bag)
        bpy.ops.object.shade_smooth()
    dm = metal_material("Dumpster", (0.015, 0.035, 0.02, 1), 0.7)
    _box("Dumpster", (5.9, -3.5, 0.75), (1.6, 2.4, 1.2), dm)
    _box("Dumpster2", (-6.0, -61.0, 0.75), (1.6, 2.4, 1.2), dm)
    return lamps
