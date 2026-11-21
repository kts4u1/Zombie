"""
Pose math for the MakeHuman 'default' rig.

Convention: every bone rotation R is expressed in *armature rest axes*
(X = character's left->right lateral axis, Y = backward, Z = up; the
character faces -Y).  Accumulated delta D_b = D_parent * R_b.  The armature
object stays at the world origin, so armature space == world space and the
root bone carries the world translation/rotation of the character.

All poses are baked to f-curves (one key per frame).
"""
import math
import numpy as np
from mathutils import Matrix, Quaternion, Vector, Euler

X = Vector((1, 0, 0))
Y = Vector((0, 1, 0))
Z = Vector((0, 0, 1))


def q_axis(axis, deg):
    return Quaternion(Vector(axis).normalized(), math.radians(deg))


def q_euler(x=0.0, y=0.0, z=0.0, order="XYZ"):
    return Euler((math.radians(x), math.radians(y), math.radians(z)), order).to_quaternion()


def frame_rot(a1, b1, a2, b2):
    """rotation mapping orthonormal frame (a1, b1) onto (a2, b2)."""
    a1 = a1.normalized(); a2 = a2.normalized()
    b1 = (b1 - a1 * b1.dot(a1)).normalized()
    b2 = (b2 - a2 * b2.dot(a2)).normalized()
    c1 = a1.cross(b1); c2 = a2.cross(b2)
    M1 = Matrix((a1, b1, c1)).transposed()
    M2 = Matrix((a2, b2, c2)).transposed()
    return (M2 @ M1.transposed()).to_quaternion()


class RigPose:
    def __init__(self, arm_ob):
        self.ob = arm_ob
        self.bones = arm_ob.data.bones
        self.order = []
        def walk(b):
            self.order.append(b.name)
            for c in b.children:
                walk(c)
        for b in self.bones:
            if b.parent is None:
                walk(b)
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in self.bones}
        self.h_rest = {b.name: Vector(b.head_local) for b in self.bones}
        self.t_rest = {b.name: Vector(b.tail_local) for b in self.bones}
        self.B = {b.name: b.matrix_local.to_quaternion() for b in self.bones}
        self.Binv = {k: v.inverted() for k, v in self.B.items()}
        self.frames = {}   # frame -> (rot dict, root loc)

    # ---------------------------------------------------------------
    def new_pose(self):
        return {"R": {}, "root_loc": Vector((0, 0, 0))}

    def fk(self, pose):
        R = pose["R"]
        D, H = {}, {}
        for b in self.order:
            p = self.parent[b]
            r = R.get(b, Quaternion())
            if p is None:
                D[b] = r
                H[b] = self.h_rest[b] + pose["root_loc"]
            else:
                D[b] = D[p] @ r
                H[b] = H[p] + D[p] @ (self.h_rest[b] - self.h_rest[p])
        pose["D"], pose["H"] = D, H
        return pose

    def point(self, pose, bone, rest_point):
        """posed position of a rest-space point rigidly attached to bone."""
        return pose["H"][bone] + pose["D"][bone] @ (Vector(rest_point) - self.h_rest[bone])

    def tail(self, pose, bone):
        return self.point(pose, bone, self.t_rest[bone])

    def set_world(self, pose, bone, Dworld):
        """choose R so that the bone's accumulated delta equals Dworld."""
        p = self.parent[bone]
        Dp = pose["D"][p] if p else Quaternion()
        pose["R"][bone] = Dp.inverted() @ Dworld
        # keep cached D consistent for children computed later
        pose["D"][bone] = Dworld

    # ---------------------------------------------------------------
    def two_bone(self, pose, upper, upper2, lower, lower2, end, target, pole, end_rot=None,
                 soften=0.999):
        """Analytic IK: upper(+upper2) / lower(+lower2) chain, end = foot/wrist bone.
        target: desired world position of `end` head.  pole: world direction the
        middle joint should point to.  Requires pose['D'],pose['H'] from fk()."""
        hr = self.h_rest
        A0, K0, E0 = hr[upper], hr[lower], hr[end]
        L1 = (K0 - A0).length
        L2 = (E0 - K0).length
        A = pose["H"][upper]
        T = Vector(target)
        d_vec = T - A
        d = d_vec.length
        d = max(min(d, (L1 + L2) * soften), abs(L1 - L2) + 1e-4)
        u = d_vec.normalized()
        v = Vector(pole) - u * Vector(pole).dot(u)
        if v.length < 1e-6:
            v = u.orthogonal()
        v.normalize()
        cos_a = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
        cos_a = max(-1.0, min(1.0, cos_a))
        sin_a = math.sqrt(1 - cos_a * cos_a)
        K = A + L1 * (cos_a * u + sin_a * v)
        E = A + u * d
        # rest pole: average rest bend direction (perpendicular part)
        r1 = (K0 - A0).normalized()
        r2 = (E0 - K0).normalized()
        rest_pole = self.rest_pole.get(upper)
        D1 = frame_rot(r1, rest_pole, (K - A).normalized(), v)
        D2 = frame_rot(r2, rest_pole, (E - K).normalized(), v)
        self.set_world(pose, upper, D1)
        self.set_world(pose, upper2, D1)
        self.set_world(pose, lower, D2)
        self.set_world(pose, lower2, D2)
        if end_rot is not None:
            self.set_world(pose, end, end_rot)
        else:
            self.set_world(pose, end, D2)
        return K, E

    def limb_points(self, pose, upper, upper2, lower, lower2, end, mid_target, end_target, pole_hint, end_rot=None):
        """aim upper segment at mid_target and lower segment at end_target."""
        hr = self.h_rest
        A0, K0, E0 = hr[upper], hr[lower], hr[end]
        L1 = (K0 - A0).length
        A = pose["H"][upper]
        dir1 = (Vector(mid_target) - A).normalized()
        K = A + dir1 * L1
        dir2 = (Vector(end_target) - K).normalized()
        r1 = (K0 - A0).normalized(); r2 = (E0 - K0).normalized()
        rp = self.rest_pole.get(upper)
        D1 = frame_rot(r1, rp, dir1, Vector(pole_hint))
        D2 = frame_rot(r2, rp, dir2, Vector(pole_hint))
        self.set_world(pose, upper, D1); self.set_world(pose, upper2, D1)
        self.set_world(pose, lower, D2); self.set_world(pose, lower2, D2)
        self.set_world(pose, end, end_rot if end_rot is not None else D2)

    def setup_poles(self):
        """rest bend directions for legs (knee forward = -Y) and arms (elbow back = +Y)."""
        self.rest_pole = {}
        for s in ("L", "R"):
            self.rest_pole["upperleg01." + s] = Vector((0, -1, 0))
            self.rest_pole["upperarm01." + s] = Vector((0, 1, 0))

    # ---------------------------------------------------------------
    def store(self, frame, pose):
        self.frames[frame] = ({k: v.copy() for k, v in pose["R"].items()}, pose["root_loc"].copy())

    def bake(self, action_name=None):
        """write stored frames as f-curves."""
        ob = self.ob
        if ob.animation_data is None:
            ob.animation_data_create()
        act = bpy_new_action(action_name or ob.name + "_act")
        frames = sorted(self.frames)
        bones = set()
        for f in frames:
            bones.update(self.frames[f][0].keys())
        root = self.order[0]
        bones.add(root)
        for b in bones:
            Binv, B = self.Binv[b], self.B[b]
            qs = []
            prev = None
            for f in frames:
                R = self.frames[f][0].get(b, Quaternion())
                q = Binv @ R @ B
                if prev is not None and prev.dot(q) < 0:
                    q = -q
                prev = q
                qs.append(q)
            path = f'pose.bones["{b}"].rotation_quaternion'
            for i in range(4):
                fc = act.fcurves.new(path, index=i, action_group=b)
                fc.keyframe_points.add(len(frames))
                arr = np.empty(len(frames) * 2)
                arr[0::2] = frames
                arr[1::2] = [q[i] for q in qs]
                fc.keyframe_points.foreach_set("co", arr)
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
        # root translation (bone local)
        path = f'pose.bones["{root}"].location'
        Binv = self.Binv[root]
        locs = [Binv @ self.frames[f][1] for f in frames]
        for i in range(3):
            fc = act.fcurves.new(path, index=i, action_group=root)
            fc.keyframe_points.add(len(frames))
            arr = np.empty(len(frames) * 2)
            arr[0::2] = frames
            arr[1::2] = [l[i] for l in locs]
            fc.keyframe_points.foreach_set("co", arr)
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        ob.animation_data.action = act
        if hasattr(ob.animation_data, "action_slot") and len(getattr(act, "slots", [])):
            ob.animation_data.action_slot = act.slots[0]
        return act


def bpy_new_action(name):
    import bpy
    return bpy.data.actions.new(name)
