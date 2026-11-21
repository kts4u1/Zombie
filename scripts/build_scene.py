"""
Builds the complete 30 s zombie shot (characters, animation, environment,
FX, 13 camera shots, render + compositor settings) and saves a .blend.

usage: python build_scene.py <assets_dir> <out.blend> [preview]
"""
import math
import os
import pickle
import sys

import bpy
from mathutils import Vector, Quaternion, Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characters
import anim
import environment
import fx
from anim import Noise, keys, sstep, clamp, lerp, FPS, NFRAMES


def T(f):
    return (f - 1) / FPS


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------
def look_quat(cam_pos, target, roll=0.0):
    d = (Vector(target) - Vector(cam_pos)).normalized()
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Quaternion((0, 0, 1), math.radians(roll))
    return q


class ShotCam:
    def __init__(self, name, lens, fstop=None, sensor=36):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = sensor
        cd.clip_start = 0.05
        cd.clip_end = 400
        if fstop:
            cd.dof.use_dof = True
            cd.dof.aperture_fstop = fstop
        self.ob = bpy.data.objects.new(name, cd)
        bpy.context.scene.collection.objects.link(self.ob)
        self.ob.rotation_mode = "QUATERNION"
        self.cd = cd

    def key(self, f, pos, target, roll=0.0, focus=None, lens=None):
        self.ob.location = pos
        self.ob.rotation_quaternion = look_quat(pos, target, roll)
        self.ob.keyframe_insert("location", frame=f)
        self.ob.keyframe_insert("rotation_quaternion", frame=f)
        if self.cd.dof.use_dof:
            self.cd.dof.focus_distance = focus if focus else (Vector(target) - Vector(pos)).length
            self.cd.dof.keyframe_insert("focus_distance", frame=f)
        if lens:
            self.cd.lens = lens
            self.cd.keyframe_insert("lens", frame=f)


def shake(n, t, amp, freq=1.0):
    return Vector((n(t, freq) * amp, n(t + 31.7, freq) * amp, n(t + 77.1, freq) * amp * 0.6))


def build_cameras(ch, events):
    ev = events["frames"]
    n = Noise(99)
    V = lambda f, k: Vector(ev[max(1, min(NFRAMES, f))][k])
    shots = []

    def shot(name, f0, f1, lens, fstop, fn, fill=0.0, fill_off=(0.6, -0.3, 0.4)):
        c = ShotCam(name, lens, fstop)
        for f in range(f0 - 1, f1 + 2):
            r = fn(f, T(f))
            c.key(f, *r[:2], **(r[2] if len(r) > 2 else {}))
        shots.append((f0, c.ob))
        if fill > 0:
            # soft cinematic fill riding with the camera, only on during this shot
            ld = bpy.data.lights.new(name + "_fill", "AREA")
            ld.size = 1.2
            ld.color = (0.78, 0.84, 1.0)
            lo = bpy.data.objects.new(name + "_fill", ld)
            bpy.context.scene.collection.objects.link(lo)
            lo.parent = c.ob
            lo.location = fill_off
            for f, e in ((1, 0.0), (f0, fill), (f1 + 1, 0.0)):
                ld.energy = e
                ld.keyframe_insert("energy", frame=f)
            for kp in ld.animation_data.action.fcurves[0].keyframe_points:
                kp.interpolation = "CONSTANT"
        return c

    vg = ch.vy_ground
    # S1: zombie idle, front 3/4 low angle, slow push-in
    def s1(f, t):
        u = t / 4.2
        p = Vector((lerp(2.9, 1.75, u), lerp(-1.6, -0.85, u), lerp(1.05, 1.25, u))) + shake(n, t, 0.012, 0.4)
        tg = V(f, "z_head") + Vector((0, 0, -0.05))
        return p, tg
    shot("S01_idle", 1, 100, 38, 2.8, s1, fill=25)

    # S2: over the zombie's shoulder, telephoto on the far victim with the phone
    def s2(f, t):
        p = Vector((0.42, 0.62, 1.74)) + shake(n, t, 0.006, 0.5)
        tg = V(f, "v_head") + Vector((0, 0, -0.25))
        return p, tg, dict(focus=(tg - p).length)
    shot("S02_target", 101, 132, 135, 2.8, s2)

    # S3: close-up, the head snaps toward camera (= toward the prey)
    def s3(f, t):
        p = Vector((0.22, -1.05, 1.66)) + shake(n, t, 0.008, 0.8)
        tg = V(f, "z_head") + Vector((0, -0.05, 0.0))
        return p, tg
    shot("S03_lock", 133, 163, 52, 2.2, s3, fill=18)

    # S4: wide low side: crouch and explosive launch, panning
    def s4(f, t):
        p = Vector((4.2, -4.2, 0.45)) + shake(n, t, 0.01, 0.6)
        tg = V(f, "z_root") + Vector((0, -0.6, 0.15))
        return p, tg
    shot("S04_launch", 164, 196, 26, None, s4, fill=60, fill_off=(0.0, 0.5, 1.2))

    # S5: front tracking, camera retreats, zombie gains on it
    def s5(f, t):
        zr = V(f, "z_root")
        u = sstep(T(197), T(254), t)
        d = lerp(6.5, 1.6, u)
        p = Vector((0.55, zr.y - d, 0.95)) + shake(n, t, 0.05, 2.5)
        tg = V(f, "z_head") + Vector((0, 0, -0.25))
        return p, tg, dict(roll=6 * n(t, 1.5))
    shot("S05_charge", 197, 254, 30, None, s5)

    # S6: side tracking wide, lamps whizzing by
    def s6(f, t):
        zr = V(f, "z_root")
        lag = lerp(0.6, 3.0, sstep(T(255), T(288), t))
        p = Vector((3.6, zr.y + lag, 1.05)) + shake(n, t, 0.04, 2.0)
        tg = zr + Vector((0, -1.0, 0.2))
        return p, tg, dict(roll=-4 + 3 * n(t, 1.2))
    shot("S06_track", 255, 288, 22, None, s6)

    # S7: the victim looks up from his phone - medium close-up
    def s7(f, t):
        vh = V(f, "v_head")
        p = Vector((0.85, vh.y + 1.35, 1.5)) + shake(n, t, 0.01, 0.9)
        tg = vh + Vector((0, 0, -0.1))
        return p, tg
    shot("S07_victim", 289, 316, 45, 2.0, s7, fill=10)

    # S8: wide 3/4: the leap and the tackle
    def s8(f, t):
        mid = (V(f, "z_root") + V(f, "v_root")) * 0.5
        imp = 1.0 if t > ch.T_IMPACT else 0.0
        p = Vector((3.3, ch.vy_imp + 1.0, 1.0)) + shake(n, t, 0.015 + 0.07 * imp * math.exp(-(t - ch.T_IMPACT) * 4 if imp else 0), 3.0)
        tg = mid + Vector((0, 0, -0.15))
        return p, tg
    shot("S08_tackle", 317, 350, 24, None, s8)

    # S9: ground level side: pin, lunge, bite, tear
    def s9(f, t):
        b = V(f, "v_bite")
        b0 = V(395, "v_bite")
        p = Vector((b0.x - 1.55, b0.y - 0.35, 0.42)) + shake(n, t, 0.012, 1.2)
        tg = V(f, "z_mouth") * 0.6 + b * 0.4 + Vector((0, 0.15, 0.08))
        return p, tg
    shot("S09_bite", 351, 422, 30, 2.8, s9, fill=14)

    # S10: close-up profile of the zombie chewing
    def s10(f, t):
        m = V(f, "z_mouth")
        base = V(470, "z_mouth")
        p = Vector((base.x - 0.80, base.y - 0.45, base.z - 0.02)) + shake(n, t, 0.006, 0.7)
        tg = m + Vector((0, 0.05, 0.04))
        return p, tg, dict(focus=(m - p).length)
    shot("S10_chew", 423, 518, 42, 2.8, s10, fill=6, fill_off=(0.3, 0.3, 0.2))

    # S11: high angle: second bite, the victim weakening, blood pool spreading
    def s11(f, t):
        b = V(f, "v_bite")
        u = sstep(T(519), T(610), t)
        p = Vector((lerp(1.9, 1.4, u), b.y + lerp(1.6, 1.2, u), lerp(2.6, 2.2, u))) + shake(n, t, 0.01, 0.5)
        return p, b + Vector((0, 0.25, 0.1))
    shot("S11_overhead", 519, 610, 30, 4.0, s11)

    # S12: extreme close-up of the jaw grinding
    def s12(f, t):
        m = V(f, "z_mouth")
        base = V(650, "z_mouth")
        p = Vector((base.x - 0.62, base.y - 0.40, base.z + 0.0)) + shake(n, t, 0.004, 0.8)
        tg = m + Vector((0, 0.03, 0.05))
        return p, tg, dict(focus=(m - p).length)
    shot("S12_jaw", 611, 682, 45, 2.8, s12, fill=4, fill_off=(0.25, 0.25, 0.15))

    # S13: final wide - the zombie hunched over the body under the flickering lamp
    def s13(f, t):
        b = V(683, "v_bite")
        u = sstep(T(683), T(720), t)
        p = Vector((-2.6, b.y - lerp(9.5, 8.6, u), 1.35)) + shake(n, t, 0.006, 0.4)
        return p, b + Vector((0, 0.5, 0.25))
    shot("S13_end", 683, 720, 32, None, s13)

    sc = bpy.context.scene
    sc.timeline_markers.clear()
    for f0, ob in shots:
        m = sc.timeline_markers.new(ob.name, frame=f0)
        m.camera = ob
    sc.camera = shots[0][1]
    return shots


# ---------------------------------------------------------------------------
def render_settings(preview=False):
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, NFRAMES
    sc.render.fps = FPS
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = 6
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.05
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_prefilter = "FAST"
    try:
        cy.denoising_quality = "BALANCED"
    except (AttributeError, TypeError):
        pass
    cy.max_bounces = 3
    cy.diffuse_bounces = 1
    cy.glossy_bounces = 2
    cy.transmission_bounces = 4
    cy.transparent_max_bounces = 12
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.sample_clamp_indirect = 4.0
    cy.blur_glossy = 1.0
    cy.use_light_tree = True
    sc.render.use_persistent_data = True
    sc.render.resolution_x = 640 if preview else 960
    sc.render.resolution_y = 360 if preview else 540
    # emissive meshes (windows, lamp bulbs) light the scene only via indirect
    # hits; the real light comes from the spot lights -> far cheaper sampling
    for m in bpy.data.materials:
        if m.name.startswith(("Building", "Lamp")) or m.name.endswith("_glow"):
            m.cycles.emission_sampling = "NONE"
        if m.node_tree:
            for nd in m.node_tree.nodes:
                if nd.type == "BSDF_PRINCIPLED":
                    nd.subsurface_method = "BURLEY"
    sc.render.resolution_percentage = 100
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    sc.cycles.motion_blur_position = "CENTER"
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 1.2
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_depth = "8"
    sc.view_layers[0].use_pass_mist = True


def compositor():
    sc = bpy.context.scene
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    # mist -> atmospheric haze with warm sodium-polluted colour
    fog_amt = nt.nodes.new("CompositorNodeMath")
    fog_amt.operation = "MULTIPLY"
    fog_amt.inputs[1].default_value = 0.45
    nt.links.new(rl.outputs["Mist"], fog_amt.inputs[0])
    fog = nt.nodes.new("CompositorNodeMixRGB")
    fog.blend_type = "MIX"
    fog.inputs[2].default_value = (0.055, 0.045, 0.04, 1)
    nt.links.new(fog_amt.outputs[0], fog.inputs[0])
    nt.links.new(rl.outputs["Image"], fog.inputs[1])
    glare = nt.nodes.new("CompositorNodeGlare")
    glare.glare_type = "FOG_GLOW"
    glare.quality = "MEDIUM"
    glare.threshold = 0.9
    glare.size = 8
    nt.links.new(fog.outputs[0], glare.inputs["Image"])
    lens = nt.nodes.new("CompositorNodeLensdist")
    lens.inputs["Distortion"].default_value = -0.012
    lens.inputs["Dispersion"].default_value = 0.012
    nt.links.new(glare.outputs[0], lens.inputs["Image"])
    # colour grade: teal shadows, warm highlights
    cb = nt.nodes.new("CompositorNodeColorBalance")
    cb.correction_method = "LIFT_GAMMA_GAIN"
    cb.lift = (0.97, 1.0, 1.03)
    cb.gamma = (1.0, 0.99, 0.98)
    cb.gain = (1.04, 1.0, 0.94)
    nt.links.new(lens.outputs["Image"], cb.inputs["Image"])
    # vignette
    ell = nt.nodes.new("CompositorNodeEllipseMask")
    ell.width, ell.height = 0.95, 0.85
    blur = nt.nodes.new("CompositorNodeBlur")
    blur.filter_type = "GAUSS"
    blur.use_relative = True
    blur.factor_x = blur.factor_y = 0.25
    blur.size_x = blur.size_y = 200
    nt.links.new(ell.outputs["Mask"], blur.inputs["Image"])
    vig = nt.nodes.new("CompositorNodeMixRGB")
    vig.blend_type = "MULTIPLY"
    vig.inputs[0].default_value = 1.0
    nt.links.new(cb.outputs["Image"], vig.inputs[1])
    ramp = nt.nodes.new("CompositorNodeMapRange")
    ramp.inputs["To Min"].default_value = 0.35
    ramp.inputs["To Max"].default_value = 1.0
    nt.links.new(blur.outputs[0], ramp.inputs["Value"])
    nt.links.new(ramp.outputs[0], vig.inputs[2])
    comp = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(vig.outputs[0], comp.inputs["Image"])


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    assets, out = argv[0], argv[1]
    preview = len(argv) > 2 and argv[2] == "preview"
    D = os.path.join(assets, "addons/mpfb/data")
    A = os.path.join(assets, "assets")
    PH = os.path.join(assets, "ph")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    z = characters.Character("zombie", D, A).build()
    v = characters.Character("victim", D, A).build()
    ch = anim.Choreo(z, v)
    ch.run()
    environment.build(PH)
    fx.build(ch, z, v)
    build_cameras(ch, ch.events)
    render_settings(preview)
    compositor()
    with open(os.path.splitext(out)[0] + "_events.pkl", "wb") as fh:
        pickle.dump(ch.events, fh)
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("SAVED", out)


if __name__ == "__main__":
    main()
