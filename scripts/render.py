"""Render a frame range of the built scene to PNGs (resumable).
usage: python render.py -- scene.blend start end outdir [threads]"""
import os
import sys
import time
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
blend, start, end, outdir = argv[0], int(argv[1]), int(argv[2]), argv[3]
threads = int(argv[4]) if len(argv) > 4 else 0
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
if threads:
    sc.render.threads_mode = "FIXED"
    sc.render.threads = threads
os.makedirs(outdir, exist_ok=True)
for f in range(start, end + 1):
    path = os.path.join(outdir, "f%04d.png" % f)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    sc.frame_set(f)
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print("RENDERED", f, round(time.time() - t, 1), flush=True)
