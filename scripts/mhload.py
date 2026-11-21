"""
Data-only loader for MakeHuman CC0 assets (base mesh, morph targets, rig,
skin weights, .mhclo proxies and .mhmat materials).

No MakeHuman / MPFB code is executed: everything here parses the CC0 data
files directly and builds Blender objects with bpy.

Coordinate convention: MakeHuman files are Y-up decimetres; Blender is
Z-up metres.  mh -> blender: (x, y, z) -> (x, -z, y) * 0.1
"""
import gzip
import json
import os
import re

import numpy as np
import bpy
import bmesh
from mathutils import Matrix, Vector

SCALE = 0.1


def mh2bl(a):
    a = np.asarray(a, dtype=np.float64)
    out = np.empty_like(a)
    out[..., 0] = a[..., 0]
    out[..., 1] = -a[..., 2]
    out[..., 2] = a[..., 1]
    return out * SCALE


# --------------------------------------------------------------------------
# OBJ
# --------------------------------------------------------------------------
class ObjData:
    def __init__(self, path):
        verts, uvs, faces, fuvs, fgroups = [], [], [], [], []
        group = None
        groups = []
        gidx = {}
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("vt "):
                    p = line.split()
                    uvs.append([float(p[1]), float(p[2])])
                elif line.startswith("f "):
                    fv, ft = [], []
                    for tok in line.split()[1:]:
                        s = tok.split("/")
                        fv.append(int(s[0]) - 1)
                        ft.append(int(s[1]) - 1 if len(s) > 1 and s[1] else -1)
                    faces.append(fv)
                    fuvs.append(ft)
                    fgroups.append(gidx.get(group, -1))
                elif line.startswith("g "):
                    group = line.split(None, 1)[1].strip()
                    if group not in gidx:
                        gidx[group] = len(groups)
                        groups.append(group)
                        fgroups_fix = None
        self.verts = np.array(verts, dtype=np.float64)  # MH coords
        self.uvs = np.array(uvs, dtype=np.float64) if uvs else np.zeros((0, 2))
        self.faces = faces
        self.fuvs = fuvs
        self.fgroups = np.array(fgroups, dtype=np.int32)
        self.groups = groups

    def group_vertex_indices(self, name):
        gi = self.groups.index(name)
        s = set()
        for f, g in zip(self.faces, self.fgroups):
            if g == gi:
                s.update(f)
        return np.array(sorted(s), dtype=np.int64)


def load_target(path):
    op = gzip.open if path.endswith(".gz") else open
    idx, d = [], []
    with op(path, "rt") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4 or line.startswith("#"):
                continue
            idx.append(int(p[0]))
            d.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(idx, dtype=np.int64), np.array(d, dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------
# Macro modifiers (same interpolation scheme MakeHuman uses)
# --------------------------------------------------------------------------
def _pair(v, lo_name, mid_name, hi_name):
    """weights for a 3-level variable with value v in [0,1] (0.5 = mid)."""
    if v < 0.5:
        t = v / 0.5
        return {lo_name: 1 - t, mid_name: t}
    t = (v - 0.5) / 0.5
    return {mid_name: 1 - t, hi_name: t}


def macro_targets(gender=0.5, age=0.5, muscle=0.5, weight=0.5, height=0.5,
                  proportions=0.5, race=None):
    """Return list of (relative target path, weight)."""
    race = race or {"african": 1 / 3, "asian": 1 / 3, "caucasian": 1 / 3}
    gw = {"female": 1 - gender, "male": gender}
    if age < 0.1875:
        t = age / 0.1875
        aw = {"baby": 1 - t, "child": t}
    elif age < 0.5:
        t = (age - 0.1875) / (0.5 - 0.1875)
        aw = {"child": 1 - t, "young": t}
    else:
        t = (age - 0.5) / 0.5
        aw = {"young": 1 - t, "old": t}
    mw = _pair(muscle, "minmuscle", "averagemuscle", "maxmuscle")
    ww = _pair(weight, "minweight", "averageweight", "maxweight")
    out = []
    for g, a in gw.items():
        for ag, b in aw.items():
            for r, c in race.items():
                w = a * b * c
                if w > 1e-4:
                    out.append((f"macrodetails/{r}-{g}-{ag}.target.gz", w))
            for m, cm in mw.items():
                for wt, cw in ww.items():
                    w = a * b * cm * cw
                    if w > 1e-4:
                        out.append((f"macrodetails/universal-{g}-{ag}-{m}-{wt}.target.gz", w))
                    if abs(height - 0.5) > 1e-3:
                        hn = "maxheight" if height > 0.5 else "minheight"
                        hw = w * abs(height - 0.5) * 2
                        if hw > 1e-4:
                            out.append((f"macrodetails/height/{g}-{ag}-{m}-{wt}-{hn}.target.gz", hw))
                    if ag != "baby" and abs(proportions - 0.5) > 1e-3:
                        pn = "idealproportions" if proportions > 0.5 else "uncommonproportions"
                        pw = w * abs(proportions - 0.5) * 2
                        if pw > 1e-4:
                            out.append((f"macrodetails/proportions/{g}-{ag}-{m}-{wt}-{pn}.target.gz", pw))
    return out


# --------------------------------------------------------------------------
# Human builder
# --------------------------------------------------------------------------
class Human:
    """Holds shaped base coordinates (MH space) for one character."""

    def __init__(self, data_dir, sys_assets_dir, name):
        self.data_dir = data_dir
        self.assets = sys_assets_dir
        self.name = name
        self.obj = ObjData(os.path.join(data_dir, "3dobjs", "base.obj"))
        self.coords = self.obj.verts.copy()  # MH coords, shaped
        self._target_cache = {}

    def apply_target(self, rel, w):
        if abs(w) < 1e-5:
            return
        path = os.path.join(self.data_dir, "targets", rel)
        if not os.path.exists(path):
            print("missing target", rel)
            return
        if path not in self._target_cache:
            self._target_cache[path] = load_target(path)
        idx, d = self._target_cache[path]
        if len(idx):
            self.coords[idx] += d * w

    def apply_macros(self, **kw):
        for rel, w in macro_targets(**kw):
            self.apply_target(rel, w)

    # ---- position helpers (blender space, after ground offset) ----
    def finalize(self):
        """Compute blender-space coordinates with feet on z=0."""
        bl = mh2bl(self.coords)
        body_idx = self.obj.group_vertex_indices("body")
        self.body_idx = body_idx
        zmin = bl[body_idx, 2].min()
        self.offset = np.array([0.0, 0.0, -zmin])
        self.bl = bl + self.offset

    def cube_center(self, cube_name):
        idx = self.obj.group_vertex_indices(cube_name)
        return self.bl[idx].mean(axis=0)

    def ref_pos(self, spec):
        s = spec["strategy"]
        if s == "CUBE":
            return self.cube_center(spec["cube_name"])
        if s == "VERTEX":
            return self.bl[spec["vertex_index"]]
        if s == "MEAN":
            return self.bl[spec["vertex_indices"]].mean(axis=0)
        raise ValueError(s)

    # ---- build blender body mesh ----
    def build_body(self, keep_groups=("body",), delete_verts=None, name=None):
        o = self.obj
        keep = {o.groups.index(g) for g in keep_groups}
        dv = delete_verts if delete_verts is not None else set()
        sel_faces = [i for i, g in enumerate(o.fgroups)
                     if g in keep and not any(v in dv for v in o.faces[i])]
        used = sorted({v for i in sel_faces for v in o.faces[i]})
        remap = {v: i for i, v in enumerate(used)}
        verts = [tuple(self.bl[v]) for v in used]
        faces = [[remap[v] for v in o.faces[i]] for i in sel_faces]
        me = bpy.data.meshes.new(name or self.name + "_body")
        me.from_pydata(verts, [], faces)
        me.update()
        uvl = me.uv_layers.new(name="UVMap")
        loop_uv = []
        for i in sel_faces:
            for t in o.fuvs[i]:
                loop_uv.append(o.uvs[t] if t >= 0 else (0, 0))
        uvl.data.foreach_set("uv", np.array(loop_uv, dtype=np.float32).ravel())
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob["mh_index_map"] = json.dumps(used) if len(used) < 0 else ""
        self.body_used = np.array(used, dtype=np.int64)
        self.body_remap = remap
        return ob

    # ---- rig ----
    def build_armature(self, rig_json, name=None):
        rig = json.load(open(rig_json))
        arm = bpy.data.armatures.new((name or self.name) + "_rig")
        ob = bpy.data.objects.new((name or self.name) + "_rig", arm)
        bpy.context.scene.collection.objects.link(ob)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = {}
        for bn, b in rig.items():
            eb = arm.edit_bones.new(bn)
            eb.head = Vector(self.ref_pos(b["head"]))
            eb.tail = Vector(self.ref_pos(b["tail"]))
            if (eb.tail - eb.head).length < 1e-4:
                eb.tail = eb.head + Vector((0, 0, 0.01))
            eb.roll = b.get("roll", 0.0)
            ebs[bn] = eb
        for bn, b in rig.items():
            p = b.get("parent")
            if p:
                ebs[bn].parent = ebs[p]
                ebs[bn].use_connect = False
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in ob.pose.bones:
            pb.rotation_mode = "QUATERNION"
        return ob

    def body_weights(self, weights_json):
        w = json.load(open(weights_json))["weights"]
        self.weights = {}
        for bn, lst in w.items():
            arr = np.array(lst, dtype=np.float64)
            if len(arr):
                self.weights[bn] = (arr[:, 0].astype(np.int64), arr[:, 1])
        return self.weights

    def assign_body_weights(self, ob):
        for bn, (idx, ws) in self.weights.items():
            vg = ob.vertex_groups.new(name=bn)
            for i, w in zip(idx, ws):
                j = self.body_remap.get(int(i))
                if j is not None and w > 1e-4:
                    vg.add([j], float(w), "REPLACE")

    def dense_weights(self):
        """bone -> dense array over all base verts."""
        n = len(self.bl)
        out = {}
        for bn, (idx, ws) in self.weights.items():
            a = np.zeros(n)
            a[idx] = ws
            out[bn] = a
        return out


def rig_object(mesh_ob, arm_ob):
    mesh_ob.parent = arm_ob
    m = mesh_ob.modifiers.new("Armature", "ARMATURE")
    m.object = arm_ob
    m.use_deform_preserve_volume = True
    return m


# --------------------------------------------------------------------------
# Proxies (.mhclo)
# --------------------------------------------------------------------------
class Mhclo:
    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(path)
        self.refs = []      # list of (idx3, w3, offset3)
        self.scales = {}    # axis -> (v1, v2, den)
        self.delete = set()
        self.obj_file = None
        self.material = None
        self.zdepth = 50
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.zdepth = 50
        mode = None
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    if line.startswith("# ") is False:
                        pass
                    continue
                p = line.split()
                key = p[0]
                if key == "verts":
                    mode = "verts"
                    continue
                if key == "delete_verts":
                    mode = "delete"
                    continue
                numeric = all(_isnum(x) or x == "-" for x in p)
                if numeric and mode == "verts":
                    if len(p) == 1:
                        self.refs.append(((int(p[0]),) * 3, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
                    elif len(p) >= 9:
                        v = [int(x) for x in p[0:3]]
                        w = [float(x) for x in p[3:6]]
                        d = [float(x) for x in p[6:9]]
                        self.refs.append((tuple(v), tuple(w), tuple(d)))
                    continue
                if numeric and mode == "delete":
                    self._parse_delete(p)
                    continue
                if key in ("x_scale", "y_scale", "z_scale"):
                    self.scales[key[0]] = (int(p[1]), int(p[2]), float(p[3]))
                elif key == "obj_file":
                    self.obj_file = os.path.join(self.dir, p[1])
                elif key == "material":
                    self.material = os.path.join(self.dir, p[1])
                elif key == "z_depth":
                    self.zdepth = int(p[1])
                elif key == "name":
                    self.name = " ".join(p[1:])

    def _parse_delete(self, p):
        i = 0
        while i < len(p):
            if i + 2 < len(p) and p[i + 1] == "-":
                a, b = int(p[i]), int(p[i + 2])
                self.delete.update(range(a, b + 1))
                i += 3
            else:
                self.delete.add(int(p[i]))
                i += 1

    def fit(self, human):
        """Return proxy vertex coords (blender space) fitted on human."""
        c = human.coords  # MH space, shaped
        sc = np.ones(3)
        for ax, k in (("x", 0), ("y", 1), ("z", 2)):
            if ax in self.scales:
                v1, v2, den = self.scales[ax]
                sc[k] = abs(c[v1, k] - c[v2, k]) / den
        idx = np.array([r[0] for r in self.refs])
        w = np.array([r[1] for r in self.refs])
        d = np.array([r[2] for r in self.refs])
        pos = (c[idx] * w[:, :, None]).sum(axis=1) + d * sc[None, :]
        self.idx, self.w = idx, w
        return mh2bl(pos) + human.offset


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_proxy(human, mhclo_path, name, arm_ob=None, dense_w=None, offset_scale=1.0,
                inflate=0.0):
    px = Mhclo(mhclo_path)
    od = ObjData(px.obj_file)
    pos = px.fit(human)
    if len(pos) != len(od.verts):
        print("WARNING proxy vertex mismatch", name, len(pos), len(od.verts))
        n = min(len(pos), len(od.verts))
        pos = pos[:n]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(p) for p in pos], [], od.faces)
    me.update()
    if len(od.uvs):
        uvl = me.uv_layers.new(name="UVMap")
        loop_uv = []
        for ft in od.fuvs:
            for t in ft:
                loop_uv.append(od.uvs[t] if t >= 0 else (0, 0))
        uvl.data.foreach_set("uv", np.array(loop_uv, dtype=np.float32).ravel())
    for p in me.polygons:
        p.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if inflate:
        # push along normals a bit (helps clothes not to clip)
        me.calc_normals_split() if hasattr(me, "calc_normals_split") else None
        co = np.array([v.co for v in me.vertices])
        nn = np.array([v.normal for v in me.vertices])
        co += nn * inflate
        me.vertices.foreach_set("co", co.ravel())
        me.update()
    if arm_ob is not None and dense_w is not None:
        for bn, arr in dense_w.items():
            vals = (arr[px.idx] * px.w).sum(axis=1)
            nz = np.nonzero(vals > 1e-4)[0]
            if len(nz) == 0:
                continue
            vg = ob.vertex_groups.new(name=bn)
            for i in nz:
                vg.add([int(i)], float(vals[i]), "REPLACE")
        rig_object(ob, arm_ob)
    ob["mhclo"] = mhclo_path
    return ob, px


def parse_mhmat(path):
    out = {}
    d = os.path.dirname(path)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            p = line.strip().split()
            if len(p) < 2 or p[0].startswith("#"):
                continue
            k = p[0]
            if k.endswith("Texture"):
                tp = os.path.join(d, os.path.basename(p[1]))
                if not os.path.exists(tp):
                    tp = os.path.join(d, p[1])
                out[k] = tp
            elif k in ("diffuseColor", "specularColor"):
                out[k] = tuple(float(x) for x in p[1:4])
            else:
                out[k] = " ".join(p[1:])
    return out
