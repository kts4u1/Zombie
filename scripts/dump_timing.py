"""Dump foot-contact times and key story beats to JSON (for sound design).
usage: python dump_timing.py <assets_dir> <out.json>"""
import json, os, sys
import bpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import characters, anim
assets, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
D = os.path.join(assets, "addons/mpfb/data"); A = os.path.join(assets, "assets")
z = characters.Character("zombie", D, A).build()
v = characters.Character("victim", D, A).build()
ch = anim.Choreo(z, v)
data = {
    "zombie_steps": sorted([(p.t_on, p.kind) for s in "LR" for p in ch.Z.feet[s].plants if p.t_on > 0]),
    "victim_steps": sorted([(p.t_on, p.kind) for s in "LR" for p in ch.V.feet[s].plants if p.t_on > 0]),
    "T_NOTICE": ch.T_NOTICE, "T_LAUNCH": ch.T_LAUNCH, "T_LEAP": ch.T_LEAP,
    "T_IMPACT": ch.T_IMPACT, "T_LAND": ch.T_LAND, "T_V_NOTICE": ch.T_V_NOTICE,
    "BITES": [list(b) for b in ch.BITES], "SWALLOWS": list(ch.SWALLOWS), "T_SHOCK": ch.T_SHOCK,
}
json.dump(data, open(out, "w"), indent=1)
print("WROTE", out, len(data["zombie_steps"]), len(data["victim_steps"]))
