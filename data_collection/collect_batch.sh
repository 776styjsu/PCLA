#!/bin/bash

# Fail this script on errors.
set -e
set -o pipefail

SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

USAGE_STRING="usage: collect_batch.sh [-h] [-m mode] [-n steps] [-s fps] [-c] [-f path/to/Carla.sh] [-o path/to/output] [-t Town01,Town02,...] [-r]
  -h    Displays this help message.
  -m    Carla agent mode: 'tm', 'basic', or 'behavior'
  -n N  Number of steps for data collection for each map
  -s N  Frames per second for data collection
  -f    Path to CARLA script
  -o    Output directory name
  -t    Comma-separated list of towns (e.g., Town01,Town02,Town10HD)
  -c    No NPCs (vehicles/pedestrians) and all traffic lights set to green
  -r    Restart CARLA for each map
  Example: collect_batch.sh -n 1200 -s 10 -f ../../carla-0.9.15/CarlaUE4.sh -o out -t Town01,Town02"

# Default values
STEPS=1200
FPS=10
N_NPCS=40
NO_RED_LIGHT=false
OUTPUT_DIR="out_autopilot"
restart=false
CARLA_PATH=""
TOWNS=""

# Parse command-line arguments
while getopts ":hm:n:s:f:o:t:cr" opt; do
  case ${opt} in
    h)
      echo "$USAGE_STRING"
      exit 0
      ;;
    m)
      MODE="$OPTARG"
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
    t)
      TOWNS="$OPTARG"
      ;;
    c)
      N_NPCS=0
      NO_RED_LIGHT=true
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

# Default town list (excluding *_Opt variants and Town15 which may be unstable)
DEFAULT_MAPS=("Town01" "Town02" "Town03" "Town04" "Town05" "Town06" "Town07" \
              "Town10HD" "Town11" "Town12" "Town13")

# Resolve MAPS (array) and TOWNS (comma-joined string)
if [[ -z "$TOWNS" ]]; then
  MAPS=("${DEFAULT_MAPS[@]}")
  TOWNS=$(IFS=, ; echo "${MAPS[*]}")
else
  IFS=',' read -r -a MAPS <<< "$TOWNS"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
AUTOPILOT_PY="$SCRIPT_DIR/collect_with_autopilot.py"

# Ensure output dir base exists
mkdir -p "$OUTPUT_DIR"

# Build optional --no-red-light flag (truthy: 1,true,yes,on)
redlight_flag=()
case "${NO_RED_LIGHT,,}" in
  1|true|yes|on)
    redlight_flag+=(--no-red-light)
    ;;
  0|false|no|off|'')
    # leave empty: no flag passed
    ;;
  *)
    echo "Warning: NO_RED_LIGHT='${NO_RED_LIGHT}' not recognized (expected true/false). Ignoring." >&2
    ;;
esac

if $restart; then
  if [[ -z "$CARLA_PATH" ]]; then
    echo "Error: -f <CARLA_PATH> is required when using -r." >&2
    echo "$USAGE_STRING"
    exit 1
  fi
  for map in "${MAPS[@]}"; do
    echo "Running script on $map"
    run_dir="${map}-${TIMESTAMP}"
    python "$AUTOPILOT_PY" \
      --mode="$MODE" \
      --town="$map" \
      --steps="$STEPS" \
      --fps="$FPS" \
      --n-npcs="$N_NPCS" \
      "${redlight_flag[@]}" \
      --out="$OUTPUT_DIR/${run_dir}" \
      --restart-carla \
      --carla-path="$CARLA_PATH"
  done
else
  run_dir="all-${TIMESTAMP}"
  echo "Running script on towns: $TOWNS"
  python "$AUTOPILOT_PY" \
    --mode="$MODE" \
    --towns="$TOWNS" \
    --steps="$STEPS" \
    --fps="$FPS" \
    --n-npcs="$N_NPCS" \
    "${redlight_flag[@]}" \
    --out="$OUTPUT_DIR/${run_dir}"
fi
