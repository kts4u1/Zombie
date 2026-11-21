"""Build the zombie and the victim from MakeHuman CC0 data."""
import os
import numpy as np
import bpy

import mhload as mh
import shading as sh


def _hash_noise1d(x, seed=0, octaves=3):
    """cheap smooth 1D value noise for numpy arrays."""
    out = np.zeros_like(x)
    amp, freq, tot = 1.0, 1.0, 0.0
    rng = np.random.default_rng(seed)
    table = rng.random(4096)
    for _ in range(octaves):
        xf = x * freq
        i0 = np.floor(xf).astype(np.int64)
        f = xf - i0
        f = f * f * (3 - 2 * f)
        a = table[i0 % 4096]
        b = table[(i0 + 1) % 4096]
        out += amp * (a * (1 - f) + b * f)
        tot += amp
        amp *= 0.5
        freq *= 2.0
    return out / tot


def _set_attr(ob, name, values):
    me = ob.data
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, "FLOAT", "POINT")
    a.data.foreach_set("value", np.asarray(values, dtype=np.float32))


def _smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Character:
    def __init__(self, kind, data_dir, assets_dir):
        self.kind = kind
        self.D = data_dir
        self.A = assets_dir
        self.zombie = kind == "zombie"
        self.parts = []

    def build(self):
        D, A = self.D, self.A
        h = mh.Human(D, A, self.kind)
        if self.zombie:
            h.apply_macros(gender=1.0, age=0.6, muscle=0.32, weight=0.08, height=0.62,
                           proportions=0.45, race={"caucasian": 0.85, "african": 0.05, "asian": 0.1})
            T = "%s"
            for rel, w in [("cheek/l-cheek-volume-decr.target.gz", 1.0), ("cheek/r-cheek-volume-decr.target.gz", 1.0),
                           ("cheek/l-cheek-bones-incr.target.gz", 0.7), ("cheek/r-cheek-bones-incr.target.gz", 0.7),
                           ("eyes/l-eye-bag-incr.target.gz", 0.8), ("eyes/r-eye-bag-incr.target.gz", 0.8),
                           ("eyes/l-eye-push1-in.target.gz", 0.6), ("eyes/r-eye-push1-in.target.gz", 0.6),
                           ("head/head-fat-decr.target.gz", 1.0), ("head/head-age-incr.target.gz", 0.8),
                           ("neck/neck-scale-depth-decr.target.gz", 0.5) if False else ("neck/measure-neck-circ-decr.target.gz", 0.6)]:
                h.apply_target(rel, w)
        else:
            h.apply_macros(gender=1.0, age=0.42, muscle=0.55, weight=0.5, height=0.5,
                           proportions=0.6, race={"caucasian": 0.6, "asian": 0.3, "african": 0.1})
        h.finalize()
        self.h = h

        if self.zombie:
            clothes = ["clothes/male_casualsuit06/male_casualsuit06.mhclo",
                       "clothes/shoes02/shoes02.mhclo"]
            hair = "hair/short04/short04.mhclo"
            skin = "skins/old_caucasian_male/old_lightskinned_male_diffuse2.png"
        else:
            clothes = ["clothes/male_casualsuit05/male_casualsuit05.mhclo",
                       "clothes/shoes06/shoes06.mhclo"]
            hair = "hair/short02/short02.mhclo"
            skin = "skins/young_caucasian_male/young_lightskinned_male_diffuse.png"
        skin = os.path.join(A, skin)
        if not os.path.exists(skin):
            d = os.path.dirname(skin)
            pngs = [f for f in os.listdir(d) if f.endswith(".png") and "diffuse" in f]
            skin = os.path.join(d, sorted(pngs)[0])

        delete = set()
        clo_objs = []
        if not self.zombie:
            for c in clothes:
                delete |= mh.Mhclo(os.path.join(A, c)).delete
        else:
            # zombie keeps the body under the (torn) shirt; feet hidden by shoes
            delete |= mh.Mhclo(os.path.join(A, clothes[1])).delete

        body = h.build_body(delete_verts=delete, name=self.kind + "_body")
        arm = h.build_armature(os.path.join(D, "rigs/standard/rig.default.json"), name=self.kind)
        h.body_weights(os.path.join(D, "rigs/standard/weights.default.json"))
        h.assign_body_weights(body)
        mh.rig_object(body, arm)
        dense = h.dense_weights()
        self.body, self.arm, self.dense = body, arm, dense

        # ---- masks on body
        self._body_masks()
        body.data.materials.append(sh.skin_material(self.kind + "_skin", skin, zombie=self.zombie))

        # ---- proxies
        eye_tex = os.path.join(A, "eyes/materials/grey_eye.png" if self.zombie else "eyes/materials/brown_eye.png")
        eyes, _ = mh.build_proxy(h, os.path.join(A, "eyes/high-poly/high-poly.mhclo"), self.kind + "_eyes", arm, dense)
        eyes.data.materials.append(sh.eye_material(self.kind + "_eyemat", eye_tex, self.zombie))
        teeth, _ = mh.build_proxy(h, os.path.join(A, "teeth/teeth_base/teeth_base.mhclo"), self.kind + "_teeth", arm, dense)
        teeth.data.materials.append(sh.teeth_material(self.kind + "_teethmat", os.path.join(A, "teeth/teeth_base/teeth.png"), self.zombie))
        tongue, _ = mh.build_proxy(h, os.path.join(A, "tongue/tongue01/tongue01.mhclo"), self.kind + "_tongue", arm, dense)
        tm = sh.teeth_material(self.kind + "_tonguemat", os.path.join(A, "tongue/tongue01/tongue01_diffuse.png"), self.zombie)
        tongue.data.materials.append(tm)
        brows, _ = mh.build_proxy(h, os.path.join(A, "eyebrows/eyebrow001/eyebrow001.mhclo"), self.kind + "_brows", arm, dense)
        brows.data.materials.append(sh.alpha_card_material(self.kind + "_browmat", os.path.join(A, "eyebrows/eyebrow001/eyebrow001.png"),
                                                           patchy=0.35 if self.zombie else 0.0))
        lashes, _ = mh.build_proxy(h, os.path.join(A, "eyelashes/eyelashes01/eyelashes01.mhclo"), self.kind + "_lashes", arm, dense)
        lashes.data.materials.append(sh.alpha_card_material(self.kind + "_lashmat", os.path.join(A, "eyelashes/eyelashes01/eyelashes01.png"),
                                                            patchy=0.5 if self.zombie else 0.0))
        hob, hpx = mh.build_proxy(h, os.path.join(A, hair), self.kind + "_hair", arm, dense)
        hm = mh.parse_mhmat(hpx.material)
        hob.data.materials.append(sh.alpha_card_material(
            self.kind + "_hairmat", hm.get("diffuseTexture"),
            tint=(0.55, 0.5, 0.45, 1) if self.zombie else (1, 1, 1, 1),
            rough=0.62 if self.zombie else 0.5, patchy=0.0,
            dark=0.75 if self.zombie else 1.0))
        self.parts += [eyes, teeth, tongue, brows, lashes, hob]

        for i, c in enumerate(clothes):
            inflate = 0.006 if (self.zombie and i == 0) else 0.0
            cob, cpx = mh.build_proxy(h, os.path.join(A, c), self.kind + "_" + os.path.basename(c)[:-6],
                                      arm, dense, inflate=inflate)
            mm = mh.parse_mhmat(cpx.material)
            cob.data.materials.append(sh.cloth_material(self.kind + "_" + cpx.name, mm, zombie=self.zombie and i == 0,
                                                        dirt=0.9 if self.zombie else 0.25))
            self._cloth_masks(cob, i)
            clo_objs.append(cob)
        self.clothes = clo_objs
        self.parts += clo_objs

        # subdivision after deformation
        for ob in [body] + clo_objs:
            m = ob.modifiers.new("Subsurf", "SUBSURF")
            m.levels = 0
            m.render_levels = 1
        for ob in [body] + self.parts:
            ob.parent = arm
        self.eyes, self.teeth, self.tongue, self.hair = eyes, teeth, tongue, hob
        return self

    # ------------------------------------------------------------------
    def bone_head(self, name):
        return np.array(self.arm.data.bones[name].head_local)

    def mouth_point(self):
        h = self.h
        up = h.obj.group_vertex_indices("helper-upper-teeth")
        lo = h.obj.group_vertex_indices("helper-lower-teeth")
        p = np.vstack([h.bl[up], h.bl[lo]])
        c = p.mean(axis=0)
        c[1] = p[:, 1].min() - 0.012  # lips in front of teeth
        return c

    def _body_masks(self):
        ob = self.body
        co = np.array([v.co for v in ob.data.vertices])
        n = len(co)
        if self.zombie:
            # sunken dark eye sockets
            ed = np.zeros(n)
            for s in ("L", "R"):
                e = self.bone_head("eye." + s)
                d = np.linalg.norm((co - e) * np.array([1.0, 0.8, 1.3]), axis=1)
                ed = np.maximum(ed, _smooth((0.045 - d) / 0.025))
            _set_attr(ob, "eyedark", ed)
            # dried blood: around lips + streaks down chin/neck/chest
            m = self.mouth_point()
            self.mouth = m
            dl = np.linalg.norm((co - m) * np.array([0.8, 1.0, 1.2]), axis=1)
            dl2 = np.linalg.norm((co - (m + np.array([0, 0.0, -0.012]))) * np.array([0.9, 1.0, 1.1]), axis=1)
            dried = _smooth((0.024 - dl2) / 0.02) * 0.8
            lips = _smooth((0.03 - np.linalg.norm((co - m) * np.array([0.7, 1.6, 2.2]), axis=1)) / 0.015)
            _set_attr(ob, "lips", lips)
            front = co[:, 1] < (m[1] + 0.07)
            dx = co[:, 0] - m[0]
            streak_len = 0.02 + 0.22 * _hash_noise1d(dx * 70 + 3.0, seed=4) ** 3
            below = (m[2] - co[:, 2])
            width = 0.025 + 0.02 * _hash_noise1d(below * 20, seed=9)
            st = (below > -0.005) & (below < streak_len) & (np.abs(dx) < width) & front
            dried = np.maximum(dried, st * _smooth((streak_len - below) / 0.08) * 0.75)
            _set_attr(ob, "dried", dried)
            # veins stronger on temples / neck / forearms
            vein = np.zeros(n)
            hd = self.bone_head("head")
            vein += _smooth((0.16 - np.linalg.norm(co - (hd + [0, 0, 0.07]), axis=1)) / 0.08) * 0.6
            vein += _smooth((0.09 - np.linalg.norm(co - self.bone_head("neck02"), axis=1)) / 0.05)
            for s in ("L", "R"):
                vein += _smooth((0.13 - np.linalg.norm(co - self.bone_head("lowerarm02." + s), axis=1)) / 0.08)
            _set_attr(ob, "vein", np.clip(vein, 0, 1))
            # wounds: left forearm bite, right cheek gash, neck
            wound = np.zeros(n)
            spots = [(self.bone_head("lowerarm01.L") + [0.0, -0.03, 0.02], 0.035),
                     (self.bone_head("neck02") + [0.045, 0.0, 0.0], 0.03),
                     (self.mouth + [-0.045, 0.025, 0.03], 0.016)]
            for p, r in spots:
                wound = np.maximum(wound, _smooth((r - np.linalg.norm(co - p, axis=1)) / r) * 1.0)
            _set_attr(ob, "wound", wound)
            # fresh blood grows from the mouth after the bite
            _set_attr(ob, "fresh_d", np.linalg.norm((co - m) * np.array([1.0, 1.0, 0.55]), axis=1))
        else:
            # bite point: left side of the neck / trapezius
            n2 = self.bone_head("neck01")
            bp = n2 + np.array([0.055, -0.01, 0.01])
            self.bite_point = bp
            _set_attr(ob, "fresh_d", np.linalg.norm(co - bp, axis=1))

    def _cloth_masks(self, cob, idx):
        co = np.array([v.co for v in cob.data.vertices])
        if self.zombie:
            m = self.mouth
            dx = co[:, 0] - m[0]
            below = m[2] - co[:, 2]
            front = co[:, 1] < (m[1] + 0.12)
            drip = 0.20 + 0.22 * _hash_noise1d(dx * 9 + 1.0, seed=11) + 0.1 * _hash_noise1d(dx * 37 + 5.0, seed=12)
            dried = front * _smooth((0.17 - np.abs(dx + 0.03 * np.sin(below * 9))) / 0.12) * _smooth((drip - below) / 0.22) * (below > 0.08) * 0.75
            _set_attr(cob, "dried", dried)
            tear = np.zeros(len(co))
            for p in ([0.08, -0.1, 1.25], [-0.12, 0.0, 1.05], [0.0, 0.12, 1.3]):
                tear = np.maximum(tear, _smooth((0.12 - np.linalg.norm(co - np.array(p), axis=1)) / 0.12))
            _set_attr(cob, "tear", tear)
            _set_attr(cob, "fresh_d", np.linalg.norm((co - m) * np.array([1.0, 1.0, 0.5]), axis=1) + 0.05)
        else:
            bp = self.bite_point
            _set_attr(cob, "fresh_d", np.linalg.norm(co - bp, axis=1))

    def blood_radius_nodes(self):
        """all animated blood-radius value nodes of this character."""
        out = []
        for ob in [self.body] + self.clothes:
            for mat in ob.data.materials:
                nd = mat.node_tree.nodes.get("blood_radius")
                if nd:
                    out.append(nd)
        return out
