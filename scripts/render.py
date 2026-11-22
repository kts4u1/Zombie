"""Render a frame range of the built scene to PNGs (resumable, safe to run in parallel).
usage: python render.py -- scene.blend start end outdir [threads] [rev]
Several processes can share one range (e.g. one forward, one with 'rev'): each frame
is claimed with an exclusive lock file, so no frame is rendered twice."""
import os
import sys
import time
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
blend, start, end, outdir = argv[0], int(argv[1]), int(argv[2]), argv[3]
threads = int(argv[4]) if len(argv) > 4 else 0
reverse = len(argv) > 5 and argv[5] == "rev"
bpy.ops.wm.open_mainfile(filepath=blend)
sc = bpy.context.scene
if threads:
    sc.render.threads_mode = "FIXED"
    sc.render.threads = threads
os.makedirs(outdir, exist_ok=True)
frames = range(end, start - 1, -1) if reverse else range(start, end + 1)
for f in frames:
    path = os.path.join(outdir, "f%04d.png" % f)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        continue
    try:
        fd = os.open(path + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        continue
    sc.frame_set(f)
    sc.render.filepath = path
    t = time.time()
    bpy.ops.render.render(write_still=True)
    os.remove(path + ".lock")
    print("RENDERED", f, round(time.time() - t, 1), flush=True)
