#!/usr/bin/env bash
# svg_to_png.sh
#
# Converts every reproducibility/images_src/*.svg (the Task 15 conceptual
# figures for the article) to PNG in images/reproducibility-of-randomness/,
# at 2x the article's ~960 CSS px rendering width -- i.e. 1920px-wide PNGs
# -- for sharpness on Zenn. SVG stays the source of truth in this repo;
# the article links the PNG, because this project has not verified how
# Zenn handles SVG served from a repository (spec section 5 / risk table).
#
# Global Constraint: all tooling runs inside a container, never directly
# on the host. This script never calls rsvg-convert on the host itself --
# every conversion happens inside a `docker run`.
#
# python:3.11-slim (the same base image Dockerfile.base uses) ships
# neither an SVG renderer nor a CJK-capable font, so this script builds a
# small, throwaway image on top of it with librsvg2-bin (rsvg-convert)
# and fonts-noto-cjk added. That Dockerfile is generated into a scratch
# temp directory and is not committed to the repo: it is a render-only
# tool, not one of the pinned measurement images under docker/, so it
# does not need to sit alongside them permanently -- re-running this
# script always regenerates the identical image from the pins below.
#
# Every apt package pulled in is pinned to an exact version (Global
# Constraint: no unpinned dependency), matching the candidate versions
# available on python:3.11-slim (Debian trixie) at the time this script
# was written. If a reader's apt mirror no longer carries these exact
# versions, the build fails loudly instead of silently resolving to a
# different (unpinned) version.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPRO_DIR/images_src"
OUT_DIR="$(cd "$REPRO_DIR/.." && pwd)/images/reproducibility-of-randomness"
IMAGE_TAG="reprobench:svgrender"

# Pinned apt package versions (python:3.11-slim / Debian trixie, as seen
# via `apt-cache policy` when this script was written). librsvg2-bin
# provides rsvg-convert; fonts-noto-cjk provides CJK (Japanese) glyphs --
# python:3.11-slim's default fonts (DejaVu) do not cover Japanese, which
# is why this image is needed at all instead of just the base image.
LIBRSVG_PIN="librsvg2-bin=2.60.0+dfsg-1+deb13u1"
NOTO_CJK_PIN="fonts-noto-cjk=1:20240730+repack1-1"

# 1920px-wide PNGs (2x of the article's ~960 CSS px rendering width),
# via rsvg-convert's zoom factor, which scales both dimensions of each
# SVG's own viewBox uniformly -- every figure here uses viewBox width
# 960, so --zoom 2 always yields exactly 1920px-wide output regardless
# of each figure's own height.
ZOOM="2"

if [ ! -d "$SRC_DIR" ]; then
  echo "svg_to_png.sh: no such directory: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cat > "$BUILD_DIR/Dockerfile" <<EOF
FROM python:3.11-slim
RUN apt-get update \\
    && apt-get install -y --no-install-recommends ${LIBRSVG_PIN} ${NOTO_CJK_PIN} \\
    && rm -rf /var/lib/apt/lists/*
EOF

echo "svg_to_png.sh: building $IMAGE_TAG (librsvg2-bin + fonts-noto-cjk on python:3.11-slim) ..."
docker build -q -f "$BUILD_DIR/Dockerfile" -t "$IMAGE_TAG" "$BUILD_DIR" >/dev/null

echo "svg_to_png.sh: rendering $SRC_DIR/*.svg -> $OUT_DIR/*.png at --zoom $ZOOM ..."
docker run --rm \
  -v "$SRC_DIR:/images_src:ro" \
  -v "$OUT_DIR:/out" \
  "$IMAGE_TAG" \
  bash -c 'set -euo pipefail; for svg in /images_src/*.svg; do name="$(basename "$svg" .svg)"; echo "  $name.svg -> $name.png"; rsvg-convert --zoom '"$ZOOM"' --background-color=white -o "/out/$name.png" "$svg"; done'

# `docker run` above writes as root inside the container; on Linux hosts
# (and this project's own CI-equivalent containers elsewhere) that would
# leave the PNGs root-owned in a directory the reader's own clone
# created. Match the pattern already used by Makefile's `report`/
# `figures` targets: chown the output back to the invoking user.
docker run --rm -v "$OUT_DIR:/fix" "$IMAGE_TAG" chown -R "$(id -u):$(id -g)" /fix

count="$(find "$OUT_DIR" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')"
echo "svg_to_png.sh: wrote $count PNG(s) to $OUT_DIR"
