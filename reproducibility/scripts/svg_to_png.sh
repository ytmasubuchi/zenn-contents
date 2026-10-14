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
# python:3.11-slim ships neither an SVG renderer nor a CJK-capable font,
# so this script builds a small, throwaway image on top of it with
# librsvg2-bin (rsvg-convert) and fonts-noto-cjk added. That Dockerfile is
# generated into a scratch temp directory and is not committed to the
# repo: it is a render-only tool, not one of the pinned measurement
# images under docker/, so it does not need to sit alongside them
# permanently -- re-running this script always regenerates the identical
# image from the pins below.
#
# Pinning, and what it actually covers (fix round 1 -- a prior version of
# this script pinned only librsvg2-bin/fonts-noto-cjk by version while
# `FROM python:3.11-slim` was a mutable tag and every *transitively*
# pulled-in package -- libcairo2, libpango, libfreetype6, fontconfig, and
# 40 others -- arrived unpinned via plain `apt-get install`. That is
# fixed here two ways:
#   1. The base image is pinned by digest, not by the (mutable) tag
#      `3.11-slim`. The digest below is what `docker pull python:3.11-
#      slim` resolved to when this script was written; `docker pull
#      python:3.11-slim` yourself and update BASE_IMAGE_DIGEST if it
#      needs to move forward. A digest pin of a multi-arch tag resolves
#      to the correct per-architecture manifest on whichever arch builds
#      it, so this is portable across the project's aarch64/x86_64 hosts.
#   2. Every one of the 47 packages `apt-get install
#      --no-install-recommends librsvg2-bin fonts-noto-cjk` newly pulls
#      in on top of that base (captured via `dpkg-query -W` right after
#      that exact install, on the exact base digest below) is pinned to
#      the version string apt actually resolved -- not just the two
#      named directly. Debian package version strings are shared across
#      architectures for a given suite (trixie here), so this list is
#      expected to resolve identically on both of the project's CPU
#      architectures; if a mirror ever stops carrying one of these exact
#      versions, `apt-get install` fails the build loudly instead of
#      silently resolving to a different, unpinned version.
# What this does NOT pin: packages already present in the base image
# layer itself (bash, coreutils, openssl, tzdata, ...) are controlled
# only by the base image digest pin above, not listed individually here,
# since this script's own `apt-get install` step never touches them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$REPRO_DIR/images_src"
OUT_DIR="$(cd "$REPRO_DIR/.." && pwd)/images/reproducibility-of-randomness"
IMAGE_TAG="reprobench:svgrender"

# `docker pull python:3.11-slim` as of 2026-10-15 (Debian trixie).
BASE_IMAGE_DIGEST="sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce"

# The 47 packages `apt-get install --no-install-recommends librsvg2-bin
# fonts-noto-cjk` newly installs on top of the base image above, each
# pinned to the exact version apt resolved (see the comment block above).
PINNED_PKGS="
fontconfig=2.15.0-2.3
fontconfig-config=2.15.0-2.3
fonts-dejavu-core=2.37-8
fonts-dejavu-mono=2.37-8
fonts-noto-cjk=1:20240730+repack1-1
libatomic1=14.2.0-19
libbrotli1=1.1.0-2+b7
libcairo-gobject2=1.18.4-1+b1
libcairo2=1.18.4-1+b1
libdatrie1=0.2.13-3+b1
libdav1d7=1.5.1-1
libdeflate0=1.23-2
libexpat1=2.8.3-1~deb13u1
libfontconfig1=2.15.0-2.3
libfreetype6=2.13.3+dfsg-1+deb13u1
libfribidi0=1.0.16-1
libgdk-pixbuf-2.0-0=2.42.12+dfsg-4+deb13u1
libgdk-pixbuf2.0-common=2.42.12+dfsg-4+deb13u1
libglib2.0-0t64=2.84.4-3~deb13u5
libgraphite2-3=1.3.14-2+deb13u1
libharfbuzz0b=10.2.0-1+deb13u1
libjbig0=2.1-6.1+b2
libjpeg62-turbo=1:2.1.5-4
liblerc4=4.0.0+ds-5
libpango-1.0-0=1.56.3-1
libpangocairo-1.0-0=1.56.3-1
libpangoft2-1.0-0=1.56.3-1
libpixman-1-0=0.44.0-3
libpng16-16t64=1.6.48-1+deb13u6
librsvg2-2=2.60.0+dfsg-1+deb13u1
librsvg2-bin=2.60.0+dfsg-1+deb13u1
libsharpyuv0=1.5.0-0.1
libthai-data=0.1.29-2
libthai0=0.1.29-2+b1
libtiff6=4.7.0-3+deb13u3
libwebp7=1.5.0-0.1
libx11-6=2:1.8.12-1
libx11-data=2:1.8.12-1
libxau6=1:1.0.11-1
libxcb-render0=1.17.0-2+b1
libxcb-shm0=1.17.0-2+b1
libxcb1=1.17.0-2+b1
libxdmcp6=1:1.1.5-1
libxext6=2:1.3.4-1+b3
libxml2=2.12.7+dfsg+really2.9.14-2.1+deb13u3
libxrender1=1:0.9.12-1
shared-mime-info=2.4-5+b2
"

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

{
  echo "FROM python@${BASE_IMAGE_DIGEST}"
  echo "RUN apt-get update \\"
  echo "    && apt-get install -y --no-install-recommends \\"
  for pkg in $PINNED_PKGS; do
    printf '         %s \\\n' "$pkg"
  done
  echo "    && rm -rf /var/lib/apt/lists/*"
} > "$BUILD_DIR/Dockerfile"

echo "svg_to_png.sh: building $IMAGE_TAG (python@${BASE_IMAGE_DIGEST} + librsvg2-bin + fonts-noto-cjk, fully pinned) ..."
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
