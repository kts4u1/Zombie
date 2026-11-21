"""Shader helpers (Cycles) for skin, zombie skin, clothes, eyes, hair, blood."""
import os
import bpy


class NB:
    """Tiny node-builder."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.mat = mat
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def n(self, kind, loc=None, **props):
        node = self.nt.nodes.new(kind)
        if loc:
            node.location = loc
        else:
            node.location = (self.x, 0)
            self.x += 220
        for k, v in props.items():
            if k.startswith("in_"):
                key = k[3:].replace("__", " ")
                node.inputs[key].default_value = v
            else:
                setattr(node, k, v)
        return node

    def link(self, a, b):
        self.nt.links.new(a, b)


def img(path, colorspace="sRGB"):
    if not path or not os.path.exists(path):
        return None
    im = bpy.data.images.load(path, check_existing=True)
    try:
        im.colorspace_settings.name = colorspace
    except TypeError:
        pass
    return im


def tex_node(nb, path, colorspace="sRGB", uv=None):
    im = img(path, colorspace)
    if im is None:
        return None
    t = nb.n("ShaderNodeTexImage")
    t.image = im
    t.interpolation = "Smart"
    if uv is not None:
        nb.link(uv, t.inputs["Vector"])
    return t


def attr(nb, name):
    a = nb.n("ShaderNodeAttribute")
    a.attribute_type = "GEOMETRY"
    a.attribute_name = name
    return a


def mix(nb, fac, a, b, blend="MIX"):
    m = nb.n("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = blend
    m.clamp_result = True
    _set(nb, m.inputs[0], fac)
    _set(nb, m.inputs[6], a)
    _set(nb, m.inputs[7], b)
    return m.outputs[2]


def math(nb, op, a, b=0.0, clamp=False):
    m = nb.n("ShaderNodeMath")
    m.operation = op
    m.use_clamp = clamp
    _set(nb, m.inputs[0], a)
    _set(nb, m.inputs[1], b)
    return m.outputs[0]


def ramp(nb, fac, stops):
    r = nb.n("ShaderNodeValToRGB")
    cr = r.color_ramp
    # stops: list of (pos, (r,g,b,a))
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (p, c) in zip(cr.elements, stops):
        el.position = p
        el.color = c if len(c) == 4 else (*c, 1.0)
    _set(nb, r.inputs[0], fac)
    return r


def _set(nb, sock, val):
    if hasattr(val, "is_linked") or hasattr(val, "links"):
        nb.link(val, sock)
    elif val is not None:
        sock.default_value = val


def principled(nb, **kw):
    p = nb.n("ShaderNodeBsdfPrincipled")
    for k, v in kw.items():
        _set(nb, p.inputs[k.replace("_", " ")], v)
    out = nb.n("ShaderNodeOutputMaterial")
    nb.link(p.outputs[0], out.inputs["Surface"])
    return p, out


def blood_color_ramp(nb, fac):
    return ramp(nb, fac, [(0.0, (0.03, 0.002, 0.002)), (1.0, (0.22, 0.004, 0.004))])


# --------------------------------------------------------------------------
def skin_material(name, diffuse, zombie=False, fresh_value_node_name=None):
    """Realistic (or rotten) skin.  Uses mesh attributes:
       eyedark, dried, fresh_d (distance), wound, vein  (all optional)."""
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    tex = tex_node(nb, diffuse, uv=tc.outputs["UV"])
    base = tex.outputs["Color"] if tex else (0.6, 0.45, 0.4, 1)
    obj = tc.outputs["Object"]

    # fine pore / wrinkle bump
    pores = nb.n("ShaderNodeTexNoise", in_Scale=900.0, in_Detail=4.0, in_Roughness=0.7)
    nb.link(obj, pores.inputs["Vector"])
    wr = nb.n("ShaderNodeTexMusgrave") if False else None
    bump = nb.n("ShaderNodeBump", in_Strength=0.12 if not zombie else 0.25, in_Distance=0.0006)
    nb.link(pores.outputs["Fac"], bump.inputs["Height"])

    rough = 0.5
    sss_w = 0.35
    col = base
    if zombie:
        # dead, desaturated flesh with a cold grey-green/yellow cast
        hs = nb.n("ShaderNodeHueSaturation", in_Saturation=0.25, in_Value=0.36)
        _set(nb, hs.inputs["Color"], base)
        col = mix(nb, 1.0, hs.outputs["Color"], (0.74, 0.78, 0.62, 1), "MULTIPLY")
        # large-scale mottling (necrotic / livor patches)
        mott = nb.n("ShaderNodeTexNoise", in_Scale=4.0, in_Detail=10.0, in_Roughness=0.62)
        nb.link(obj, mott.inputs["Vector"])
        mr = ramp(nb, mott.outputs["Fac"], [(0.32, (0.50, 0.36, 0.44)), (0.45, (0.95, 0.93, 0.86)),
                                           (0.58, (0.85, 0.88, 0.66)), (0.72, (0.55, 0.58, 0.40))])
        col = mix(nb, 0.85, col, mr.outputs["Color"], "MULTIPLY")
        bru = nb.n("ShaderNodeTexNoise", in_Scale=2.2, in_Detail=4.0, in_Roughness=0.5)
        nb.link(obj, bru.inputs["Vector"])
        br = ramp(nb, bru.outputs["Fac"], [(0.58, (1, 1, 1)), (0.72, (0.55, 0.38, 0.50))])
        col = mix(nb, 0.9, col, br.outputs["Color"], "MULTIPLY")
        # grime / dirt smudges
        gn = nb.n("ShaderNodeTexNoise", in_Scale=11.0, in_Detail=12.0, in_Roughness=0.72)
        nb.link(obj, gn.inputs["Vector"])
        gr = ramp(nb, gn.outputs["Fac"], [(0.5, (1, 1, 1)), (0.68, (0.42, 0.36, 0.30))])
        col = mix(nb, 0.85, col, gr.outputs["Color"], "MULTIPLY")
        # cavity grime from mesh curvature
        geo = nb.n("ShaderNodeNewGeometry")
        cav = ramp(nb, geo.outputs["Pointiness"], [(0.47, (0.35, 0.30, 0.28)), (0.52, (1, 1, 1))])
        col = mix(nb, 0.8, col, cav.outputs["Color"], "MULTIPLY")
        # thin branching veins (ridged noise), only where the vein mask is set
        vn = nb.n("ShaderNodeTexNoise", in_Scale=16.0, in_Detail=4.0, in_Roughness=0.55, in_Distortion=0.6)
        nb.link(obj, vn.inputs["Vector"])
        rid = math(nb, "ABSOLUTE", math(nb, "SUBTRACT", vn.outputs["Fac"], 0.5))
        vr = ramp(nb, rid, [(0.0, (1, 1, 1)), (0.022, (0, 0, 0))])
        va = attr(nb, "vein")
        vf = math(nb, "MULTIPLY", vr.outputs["Color"], math(nb, "ADD", math(nb, "MULTIPLY", va.outputs["Fac"], 0.75), 0.08), clamp=True)
        col = mix(nb, vf, col, (0.10, 0.11, 0.17, 1), "MIX")
        # dark sunken eyes
        ea = attr(nb, "eyedark")
        col = mix(nb, math(nb, "MULTIPLY", ea.outputs["Fac"], 0.85), col, (0.09, 0.045, 0.06, 1))
        # cyanotic lips
        la = attr(nb, "lips")
        col = mix(nb, math(nb, "MULTIPLY", la.outputs["Fac"], 0.8), col, (0.16, 0.12, 0.14, 1))
        # wounds (raw flesh)
        wa = attr(nb, "wound")
        wn = nb.n("ShaderNodeTexNoise", in_Scale=60.0, in_Detail=6.0)
        nb.link(obj, wn.inputs["Vector"])
        wr = ramp(nb, wn.outputs["Fac"], [(0.3, (0.015, 0.002, 0.002)), (0.55, (0.10, 0.012, 0.01)),
                                         (0.75, (0.18, 0.09, 0.06))])
        wv = math(nb, "ADD", wa.outputs["Fac"], math(nb, "MULTIPLY", wn.outputs["Fac"], 0.4))
        wmask = ramp(nb, wv, [(0.6, (0, 0, 0)), (0.68, (1, 1, 1))]).outputs["Color"]
        col = mix(nb, wmask, col, wr.outputs["Color"])
        b2 = nb.n("ShaderNodeBump", in_Strength=0.6, in_Distance=0.004)
        nb.link(math(nb, "MULTIPLY", wmask, -1.0), b2.inputs["Height"])
        nb.link(bump.outputs["Normal"], b2.inputs["Normal"])
        bump = b2
        rough = math(nb, "ADD", math(nb, "MULTIPLY", mott.outputs["Fac"], 0.25), 0.33)
        rough = math(nb, "SUBTRACT", rough, math(nb, "MULTIPLY", wmask, 0.25))
        sss_w = 0.1
        # dried blood around mouth / chin / chest (soft, uneven edges)
        da = attr(nb, "dried")
        dn = nb.n("ShaderNodeTexNoise", in_Scale=30.0, in_Detail=8.0, in_Roughness=0.7)
        nb.link(obj, dn.inputs["Vector"])
        dv = math(nb, "ADD", da.outputs["Fac"], math(nb, "MULTIPLY", dn.outputs["Fac"], 0.45))
        dm = ramp(nb, dv, [(0.5, (0, 0, 0)), (0.62, (0.55, 0.55, 0.55)), (0.8, (0.95, 0.95, 0.95))]).outputs["Color"]
        dcol = ramp(nb, dn.outputs["Fac"], [(0.35, (0.014, 0.004, 0.003)), (0.65, (0.034, 0.007, 0.005))])
        col = mix(nb, dm, col, dcol.outputs["Color"])
        rough = math(nb, "SUBTRACT", rough, math(nb, "MULTIPLY", dm, 0.05))

    # fresh, wet blood: attribute fresh_d (distance from blood origin, metres)
    # compared with an animated radius value node
    fa = attr(nb, "fresh_d")
    rad = nb.n("ShaderNodeValue")
    rad.name = rad.label = "blood_radius"
    rad.outputs[0].default_value = 0.0
    fn = nb.n("ShaderNodeTexNoise", in_Scale=38.0, in_Detail=6.0)
    nb.link(obj, fn.inputs["Vector"])
    d_noisy = math(nb, "ADD", fa.outputs["Fac"], math(nb, "MULTIPLY", fn.outputs["Fac"], 0.06))
    fresh = math(nb, "LESS_THAN", d_noisy, rad.outputs[0])
    bcol = ramp(nb, fn.outputs["Fac"], [(0.35, (0.05, 0.002, 0.002)), (0.7, (0.14, 0.005, 0.004))])
    col = mix(nb, fresh, col, bcol.outputs["Color"])
    rough = math(nb, "SUBTRACT", rough if not isinstance(rough, float) else rough,
                 math(nb, "MULTIPLY", fresh, 0.38))
    sss = math(nb, "MULTIPLY", math(nb, "SUBTRACT", 1.0, fresh), sss_w)

    p, out = principled(nb, Base_Color=col, Roughness=rough,
                        **{"Subsurface Weight": sss})
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2) if not zombie else (0.6, 0.35, 0.3)
    p.inputs["Subsurface Scale"].default_value = 0.006
    p.inputs["Specular IOR Level"].default_value = 0.45
    p.inputs["Coat Weight"].default_value = 0.0
    nb.link(bump.outputs["Normal"], p.inputs["Normal"])
    cf = nb.n("ShaderNodeMath")
    cf.operation = "MULTIPLY"
    nb.link(fresh, cf.inputs[0])
    cf.inputs[1].default_value = 0.8
    nb.link(cf.outputs[0], p.inputs["Coat Weight"])
    p.inputs["Coat Roughness"].default_value = 0.08
    mat["blood_radius_node"] = rad.name
    return mat


def cloth_material(name, mhmat, zombie=False, dirt=0.0):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    t = tex_node(nb, mhmat.get("diffuseTexture"), uv=tc.outputs["UV"])
    col = t.outputs["Color"] if t else (0.5, 0.5, 0.5, 1)
    alpha = t.outputs["Alpha"] if (t and t.image.channels == 4 and "hair" in name) else 1.0
    ao = tex_node(nb, mhmat.get("aomapTexture"), "Non-Color", uv=tc.outputs["UV"])
    if ao:
        col = mix(nb, 0.8, col, ao.outputs["Color"], "MULTIPLY")
    obj = tc.outputs["Object"]
    rough = 0.85
    if zombie or dirt > 0:
        dn = nb.n("ShaderNodeTexNoise", in_Scale=6.0, in_Detail=10.0, in_Roughness=0.7)
        nb.link(obj, dn.inputs["Vector"])
        dr = ramp(nb, dn.outputs["Fac"], [(0.38, (0, 0, 0)), (0.72, (1, 1, 1))])
        col = mix(nb, math(nb, "MULTIPLY", dr.outputs["Color"], dirt), col, (0.16, 0.13, 0.09, 1), "MULTIPLY")
        col = mix(nb, dirt * 0.35, col, (0.30, 0.26, 0.2, 1), "MIX")
        col = mix(nb, dirt * 0.6, col, (0.78, 0.72, 0.62, 1), "MULTIPLY")
    if zombie:
        # old dried blood stains on garment (attribute 'dried')
        da = attr(nb, "dried")
        bn = nb.n("ShaderNodeTexNoise", in_Scale=12.0, in_Detail=8.0, in_Roughness=0.65)
        nb.link(obj, bn.inputs["Vector"])
        m = math(nb, "ADD", da.outputs["Fac"], math(nb, "MULTIPLY", bn.outputs["Fac"], 0.5))
        st = ramp(nb, m, [(0.5, (0, 0, 0)), (0.62, (0.45, 0.45, 0.45)), (0.8, (0.95, 0.95, 0.95))])
        bc = ramp(nb, bn.outputs["Fac"], [(0.3, (0.035, 0.006, 0.005)), (0.7, (0.10, 0.014, 0.010))])
        col = mix(nb, st.outputs["Color"], col, bc.outputs["Color"])
        # torn holes
        hn = nb.n("ShaderNodeTexNoise", in_Scale=4.5, in_Detail=6.0, in_Roughness=0.7)
        nb.link(obj, hn.inputs["Vector"])
        ha = attr(nb, "tear")
        hv = math(nb, "ADD", math(nb, "MULTIPLY", hn.outputs["Fac"], 0.9), math(nb, "MULTIPLY", ha.outputs["Fac"], 0.5))
        alpha = math(nb, "LESS_THAN", hv, 0.84)
        edge = math(nb, "GREATER_THAN", hv, 0.76)
        col = mix(nb, edge, col, (0.06, 0.04, 0.03, 1), "MIX")
    # fresh blood (animated)
    fa = attr(nb, "fresh_d")
    rad = nb.n("ShaderNodeValue")
    rad.name = rad.label = "blood_radius"
    rad.outputs[0].default_value = 0.0
    fn = nb.n("ShaderNodeTexNoise", in_Scale=20.0, in_Detail=6.0)
    nb.link(obj, fn.inputs["Vector"])
    fresh = math(nb, "LESS_THAN", math(nb, "ADD", fa.outputs["Fac"], math(nb, "MULTIPLY", fn.outputs["Fac"], 0.08)), rad.outputs[0])
    col = mix(nb, fresh, col, (0.09, 0.003, 0.003, 1), "MIX")
    rough = math(nb, "SUBTRACT", rough, math(nb, "MULTIPLY", fresh, 0.55))
    p, out = principled(nb, Base_Color=col, Roughness=rough, Alpha=alpha)
    p.inputs["Sheen Weight"].default_value = 0.12
    p.inputs["Specular IOR Level"].default_value = 0.3
    nm = tex_node(nb, mhmat.get("normalmapTexture"), "Non-Color", uv=tc.outputs["UV"])
    if nm:
        nmn = nb.n("ShaderNodeNormalMap", in_Strength=1.0)
        nb.link(nm.outputs["Color"], nmn.inputs["Color"])
        nb.link(nmn.outputs["Normal"], p.inputs["Normal"])
    return mat


def alpha_card_material(name, tex_path, tint=(1, 1, 1, 1), rough=0.45, patchy=0.0, dark=1.0):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    t = tex_node(nb, tex_path, uv=tc.outputs["UV"])
    col = mix(nb, 1.0, t.outputs["Color"], tint, "MULTIPLY")
    if dark != 1.0:
        col = mix(nb, 1.0, col, (dark, dark, dark, 1), "MULTIPLY")
    alpha = t.outputs["Alpha"]
    if patchy > 0:
        pn = nb.n("ShaderNodeTexNoise", in_Scale=9.0, in_Detail=3.0)
        nb.link(tc.outputs["Object"], pn.inputs["Vector"])
        alpha = math(nb, "MULTIPLY", alpha, math(nb, "GREATER_THAN", pn.outputs["Fac"], patchy))
    p, out = principled(nb, Base_Color=col, Roughness=rough, Alpha=alpha)
    p.inputs["Specular IOR Level"].default_value = 0.5
    return mat


def eye_material(name, tex_path, zombie=False):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    t = tex_node(nb, tex_path, uv=tc.outputs["UV"])
    col = t.outputs["Color"]
    if zombie:
        hs = nb.n("ShaderNodeHueSaturation", in_Saturation=0.3, in_Value=0.9)
        nb.link(col, hs.inputs["Color"])
        col = mix(nb, 0.18, hs.outputs["Color"], (0.70, 0.72, 0.66, 1))
        vn = nb.n("ShaderNodeTexNoise", in_Scale=120.0, in_Detail=8.0)
        nb.link(tc.outputs["Object"], vn.inputs["Vector"])
        vr = ramp(nb, vn.outputs["Fac"], [(0.55, (0, 0, 0)), (0.62, (1, 1, 1))])
        col = mix(nb, math(nb, "MULTIPLY", vr.outputs["Color"], 0.6), col, (0.45, 0.06, 0.05, 1))
        col = mix(nb, 0.6, col, (0.78, 0.70, 0.50, 1), "MULTIPLY")
    p, out = principled(nb, Base_Color=col, Roughness=0.25, Alpha=t.outputs["Alpha"])
    p.inputs["Coat Weight"].default_value = 1.0
    p.inputs["Coat Roughness"].default_value = 0.02
    p.inputs["Subsurface Weight"].default_value = 0.1
    return mat


def teeth_material(name, tex_path, zombie=False):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    tc = nb.n("ShaderNodeTexCoord")
    t = tex_node(nb, tex_path, uv=tc.outputs["UV"])
    col = t.outputs["Color"]
    if zombie:
        col = mix(nb, 1.0, col, (0.62, 0.48, 0.30, 1), "MULTIPLY")
        n = nb.n("ShaderNodeTexNoise", in_Scale=80.0, in_Detail=4.0)
        nb.link(tc.outputs["Object"], n.inputs["Vector"])
        r = ramp(nb, n.outputs["Fac"], [(0.45, (0, 0, 0)), (0.7, (1, 1, 1))])
        col = mix(nb, r.outputs["Color"], col, (0.12, 0.01, 0.01, 1))
    p, out = principled(nb, Base_Color=col, Roughness=0.2)
    p.inputs["Subsurface Weight"].default_value = 0.2
    p.inputs["Coat Weight"].default_value = 0.6
    return mat


def blood_material(name="Blood"):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    p, out = principled(nb, Base_Color=(0.045, 0.0015, 0.0012, 1), Roughness=0.08)
    p.inputs["Coat Weight"].default_value = 1.0
    p.inputs["Coat Roughness"].default_value = 0.02
    p.inputs["Subsurface Weight"].default_value = 0.0
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.1, 0.1)
    p.inputs["Subsurface Scale"].default_value = 0.002
    return mat
