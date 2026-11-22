#!/usr/bin/env bash
# Downloads everything the render needs (all CC0 / open data) into $ASSETS.
#   - Blender as a Python module (bpy 4.5 LTS, from PyPI)
#   - MakeHuman/MPFB CC0 data (base mesh, targets, rig, weights)  [data only]
#   - MakeHuman CC0 system assets (skins, eyes, teeth, hair, clothes)
#   - Poly Haven CC0 textures + night HDRI
#   - CC0 sound effects from Freesound (list in scripts/sounds.json) -> $ASSETS/snd/*.wav
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ASSETS=${ASSETS:-$(pwd)/.assets}
mkdir -p "$ASSETS"/{dl,ph}
cd "$ASSETS"

if [ ! -x venv/bin/python ]; then
  uv venv -p 3.11 venv
  VIRTUAL_ENV="$ASSETS/venv" uv pip install "bpy==4.5.*" pillow numpy
fi

if [ ! -d addons/mpfb/data ]; then
  curl -sSL -o dl/mpfb.zip "https://extensions.blender.org/download/sha256:c075b3e9b94b8e5556a5f03f4653602d02ea306de08126e8f55ed6dee28a2dad/add-on-mpfb-v2.0.18.zip"
  mkdir -p addons && unzip -q -o dl/mpfb.zip "data/*" -d addons/mpfb
fi

if [ ! -d assets/skins ]; then
  curl -sSL -C - --retry 5 -o dl/sysassets.zip \
    https://files2.makehumancommunity.org/asset_packs/makehuman_system_assets/makehuman_system_assets_cc0.zip
  mkdir -p assets && unzip -q -o dl/sysassets.zip -d assets
fi

PH=https://dl.polyhaven.org/file/ph-assets
get() { [ -f "ph/$2" ] || curl -sSL -o "ph/$2" "$1"; }
for t in asphalt_02 concrete_pavement; do
  for m in diff nor_gl rough; do get "$PH/Textures/jpg/2k/$t/${t}_${m}_2k.jpg" "${t}_${m}_2k.jpg"; done
done
get "$PH/Textures/jpg/2k/brick_wall_001/brick_wall_001_diffuse_2k.jpg" brick_wall_001_diff_2k.jpg
get "$PH/Textures/jpg/2k/brick_wall_001/brick_wall_001_nor_gl_2k.jpg" brick_wall_001_nor_gl_2k.jpg
get "$PH/Textures/jpg/2k/brick_wall_001/brick_wall_001_rough_2k.jpg" brick_wall_001_rough_2k.jpg
get "$PH/HDRIs/hdr/2k/moonless_golf_2k.hdr" moonless_golf_2k.hdr
mkdir -p snd/raw snd/wav
python3 - "$HERE/sounds.json" <<'PY' | while read -r id url; do
import json, sys
for d in json.load(open(sys.argv[1])):
    print(d["id"], d["preview"])
PY
  [ -f "snd/raw/$id.mp3" ] || curl -sSL --retry 3 -o "snd/raw/$id.mp3" "$url"
  [ -f "snd/wav/$id.wav" ] || ffmpeg -loglevel error -y -i "snd/raw/$id.mp3" -ac 2 -ar 48000 "snd/wav/$id.wav"
done
echo "assets ready in $ASSETS"
