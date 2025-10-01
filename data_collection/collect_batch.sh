#!/bin/bash

# Fail this script on errors.
set -e
set -o pipefail

USAGE_STRING="usage: collect_batch.sh [-h] [-n steps] [-s fps] [-f path/to/Carla.sh] [-o path/to/output] [-r]
  -h    Displays this help message.
  -n N  Number of steps for data collection for each map
  -s N  Frames per second for data collection
  -f N  Path to CARLA script
  -o N  Output directory name
  -r    Restart CARLA for each map
  Example: collect_batch.sh -n 1200 -s 10 -f ../../carla-0.9.15/CarlaUE4.sh -o out"

# Default values
STEPS=1200
FPS=10
OUTPUT_DIR="out_autopilot"
restart=false
CARLA_PATH=""

# Parse command-line arguments
while getopts ":hn:s:f:o:r" opt; do
  case ${opt} in
    h)
      echo "$USAGE_STRING"
      exit 0
      ;;
    n)
      STEPS="$OPTARG"
      ;;
    s)
      FPS="$OPTARG"
      ;;
    f)
      CARLA_PATH="$OPTARG"
      ;;
    o)
      OUTPUT_DIR="$OPTARG"
      ;;
    r)
      restart=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "$USAGE_STRING"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "$USAGE_STRING"
      exit 1
      ;;
  esac
done

# Town list (excluding *_Opt variants and Town15 that may cause crash)
MAPS=("Town01" "Town02" "Town03" "Town04" "Town05" "Town06" "Town07" \
      "Town10HD" "Town11" "Town12" "Town13")

ALL_TOWNS=$(IFS=, ; echo "${MAPS[*]}")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if $restart; then
  if [[ -z "$CARLA_PATH" ]]; then
    echo "Error: -f <CARLA_PATH> is required when using -r." >&2
    echo "$USAGE_STRING"
    exit 1
  fi
  for map in "${MAPS[@]}"; do
    echo "Running script on $map"
    run_dir="${map}-${TIMESTAMP}"
    python collect_with_autopilot.py \
      --town="$map" \
      --steps="$STEPS" \
      --fps="$FPS" \
      --out="$OUTPUT_DIR/${run_dir}" \
      --restart-carla \
      --carla-path="$CARLA_PATH"
  done
else
  run_dir="all-${TIMESTAMP}"
  python collect_with_autopilot.py \
    --towns="$ALL_TOWNS" \
    --steps="$STEPS" \
    --fps="$FPS" \
    --out="$OUTPUT_DIR/${run_dir}"
fi
