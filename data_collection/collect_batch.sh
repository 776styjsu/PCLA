#!/bin/bash

# Fail this script on errors.
set -e
set -o pipefail

USAGE_STRING="usage: collect_batch.sh [-h] [-n steps] [-f fps]
  -h    Displays this help message.
  -n N  Number of steps for data collection for each map
  -f N  Number of frames per second for data collection
  -o N  Output directory name
  Example: commons-lang3-3.0"

STEPS=1200
FPS=10
OUTPUT_DIR="out_autopilot"

# Parse command-line arguments
while getopts ":hvrf:ao:t:c:n:" opt; do
  case ${opt} in
    h)
      # Display help message
      echo "$USAGE_STRING"
      exit 0
      ;;
    n)
      # Number of steps
      STEPS="$OPTARG"
      ;;
    f)
      # FPS
      FPS="$OPTARG"
      ;;
    o)
      # Output dir
      OUT_DIR="$OPTARG"
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

MAPS=("Town01" "Town01_Opt" "Town02" "Town02_Opt" "Town03" "Town03_Opt" \
      "Town04" "Town04_Opt" "Town05" "Town05_Opt" "Town06" "Town07" \
      "Town10HD" "Town10HD_Opt" "Town11" "Town12" "Town13" "Town15")

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

for map in "${MAPS[@]}"; do
    echo "Running script on $map"
    run_dir="${map}-${TIMESTAMP}"
    python collect_with_autopilot.py --town="$map" --steps="$STEPS" --fps="$FPS" --out="$OUTPUT_DIR/${run_dir}"
done