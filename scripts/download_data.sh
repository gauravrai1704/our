#!/usr/bin/env bash
# Download the datasets that have stable direct links into datasets/raw/.
# The rest need a browser or Google Drive (see RUNBOOK.md, step 3b).
#
#   bash scripts/download_data.sh            # run from the repo root
#
# Re-running skips archives that are already downloaded. Check that each link
# still works; dataset hosts move files from time to time.
set -euo pipefail
RAW=datasets/raw
mkdir -p "$RAW"
cd "$RAW"

get() {  # get <url> [output name]
  local out="${2:-$(basename "$1")}"
  if [ -s "$out" ]; then echo "have $out"; return; fi
  echo "downloading $out"
  wget -c --tries=5 -O "$out.part" "$1" && mv "$out.part" "$out"
}

# DIV2K (train 800 + valid 100 HR)   https://data.vision.ee.ethz.ch/cvl/DIV2K/
get http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
get http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
# Flickr2K (2650 HR), from the EDSR authors
get https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
# SR benchmarks with the official MATLAB-bicubic LR images (Set5, Set14, B100, Urban100)
get https://cv.snu.ac.kr/research/EDSR/benchmark.tar
# Kodak24
mkdir -p Kodak24
for i in $(seq -w 1 24); do
  [ -s "Kodak24/kodim$i.png" ] || wget -q -O "Kodak24/kodim$i.png" "http://r0k.us/graphics/kodak/kodak/kodim$i.png"
done

echo "extracting"
[ -d DIV2K_train_HR ] || unzip -q DIV2K_train_HR.zip
[ -d DIV2K_valid_HR ] || unzip -q DIV2K_valid_HR.zip
[ -d Flickr2K ] || tar -xf Flickr2K.tar
[ -d benchmark ] || tar -xf benchmark.tar
echo "done: $(pwd)"
ls
