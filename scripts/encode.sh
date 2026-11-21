#!/usr/bin/env bash
# frames (960x540 PNG) + soundtrack -> 1280x720 H.264 MP4 with film grain and fades
set -euo pipefail
FRAMES=$1; AUDIO=$2; OUT=$3
mkdir -p "$(dirname "$OUT")"
ffmpeg -y -framerate 24 -i "$FRAMES/f%04d.png" -i "$AUDIO" \
  -vf "scale=1280:720:flags=lanczos,eq=gamma=1.06:contrast=1.03,noise=alls=7:allf=t+u,fade=t=in:st=0:d=0.4,fade=t=out:st=29.1:d=0.9,format=yuv420p" \
  -c:v libx264 -preset slow -crf 18 -profile:v high -movflags +faststart \
  -c:a aac -b:a 192k -shortest "$OUT"
