#!/usr/bin/env bash
# collect_batch.sh — robust batch launcher for CARLA collection

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- logging ----------
_is_tty() { [[ -t 1 ]]; }
_color()  { _is_tty && command -v tput >/dev/null && tput setaf "$1" || true; }
_norm()   { _is_tty && command -v tput >/dev/null && tput sgr0 || true; }
info()    { printf "%s[INFO]%s %s\n"  "$(_color 6)" "$(_norm)" "$*"; }
warn()    { printf "%s[WARN]%s %s\n"  "$(_color 3)" "$(_norm)" "$*" >&2; }
error()   { printf "%s[ERROR]%s %s\n" "$(_color 1)" "$(_norm)" "$*" >&2; }
die()     { error "$*"; exit 1; }

usage() {
  cat <<EOF
usage: $(basename "$0") [options] [-- extra args passed to collect_with_autopilot.py]

Core:
  -m MODE      Agent mode: tm | basic | behavior (default: basic)
  -t LIST      Towns CSV (e.g., Town01,Town02,Town10HD). If omitted, uses defaults
  -o DIR       Output directory root (default: out_autopilot)
  -r           Restart CARLA per town (requires -f PATH)
  -f PATH      Path to CARLA launcher (CarlaUE4.sh / Carla.sh)

Timing & length:
  -n N         Steps per town (default: 1200)
  -s N         FPS (default: 10)

Driving:
  -v N         Target speed km/h (basic/behavior) (default: 20)
  -M N         Min speed considered 'moving' km/h (default: 2)
  -x N         Stall patience in frames (default: 20)
  -R           Disable respawn-on-stall (enabled by default)

Scene:
  -c           Clean map: zero NPCs + set no-red-light
  --npcs N     NPC count (overrides -c's zero)

Rendering / IO:
  -W N         Camera width  (default: 320)
  -H N         Camera height (default: 180)

CARLA:
  -p PORT      Traffic Manager port (default: 8001)

Misc:
  -V           Verbose (echo full python command)
  -D           Dry-run (print commands, do not execute)
  -h           Help

Examples:
  $(basename "$0") -n 1200 -s 10 -f ../../carla-0.9.15/CarlaUE4.sh -o out -t Town01,Town02
  $(basename "$0") -m behavior -v 25 -c -r -f ~/carla/CarlaUE4.sh -t Town10HD,Town11 -W 320 -H 180
EOF
}

# ---------- defaults ----------
MODE="basic"
STEPS=1200
FPS=10
TARGET_SPEED=20
MIN_SPEED_KMH=2
STALL_PATIENCE=20
RGB_W=320
RGB_H=180
TM_PORT=8000
N_NPCS=40
NO_RED_LIGHT="false"
RESPAWN_ON_STALL="true"
RESTART_PER_TOWN="false"
OUTPUT_DIR="out_autopilot"
CARLA_PATH=""
TOWNS=""
VERBOSE="false"
DRY_RUN="false"

DEFAULT_MAPS=(Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD Town11 Town12 Town13)

SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
AUTOPILOT_PY="${SCRIPT_DIR}/collect_with_autopilot.py"

# ---------- helpers ----------
is_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

run_cmd() {
  # prints command when -V or -D; runs only if not dry-run
  local PYBIN
  PYBIN="$(command -v python3 || command -v python || true)"
  [[ -n "$PYBIN" ]] || die "python or python3 not found in PATH"

  if [[ "$VERBOSE" == "true" || "$DRY_RUN" == "true" ]]; then
    printf "[CMD] %q " "$PYBIN"
    printf "%q " -u "$AUTOPILOT_PY" "$@"
    printf "\n"
  fi
  if [[ "$DRY_RUN" == "false" ]]; then
    "$PYBIN" -u "$AUTOPILOT_PY" "$@"
  fi
}

# ---------- parse args ----------
# Manual pre-scan for the lone long option --npcs
PRE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --npcs)
      shift
      [[ $# -gt 0 ]] || die "--npcs requires a value"
      N_NPCS="$1"; shift
      ;;
    --) shift; break ;;
    *) PRE_ARGS+=("$1"); shift ;;
  esac
done
set -- "${PRE_ARGS[@]}" "$@"   # restore positional + keep post-`--` extras

while getopts ":hm:n:s:v:f:o:t:rp:W:H:M:x:VRDc" opt; do
  case "$opt" in
    h) usage; exit 0 ;;
    m) MODE="$OPTARG" ;;
    n) STEPS="$OPTARG" ;;
    s) FPS="$OPTARG" ;;
    v) TARGET_SPEED="$OPTARG" ;;
    f) CARLA_PATH="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    t) TOWNS="$OPTARG" ;;
    r) RESTART_PER_TOWN="true" ;;
    R) RESPAWN_ON_STALL="false" ;;
    c) N_NPCS=0; NO_RED_LIGHT="true" ;;
    p) TM_PORT="$OPTARG" ;;
    W) RGB_W="$OPTARG" ;;
    H) RGB_H="$OPTARG" ;;
    M) MIN_SPEED_KMH="$OPTARG" ;;
    x) STALL_PATIENCE="$OPTARG" ;;
    V) VERBOSE="true" ;;
    D) DRY_RUN="true" ;;
    \?) die "Unknown option: -$OPTARG (see -h)" ;;
    :)  die "Option -$OPTARG requires an argument (see -h)" ;;
  esac
done
shift $((OPTIND - 1))

EXTRA_ARGS=("$@")  # pass-through to Python

# ---------- validation ----------
[[ -f "$AUTOPILOT_PY" ]] || die "Missing ${AUTOPILOT_PY} (run from repo root or fix path)."
case "$MODE" in tm|basic|behavior) ;; *) die "-m must be one of: tm, basic, behavior (got '$MODE')";; esac

for pair in \
  "STEPS:$STEPS" "FPS:$FPS" "TARGET_SPEED:$TARGET_SPEED" \
  "MIN_SPEED_KMH:$MIN_SPEED_KMH" "STALL_PATIENCE:$STALL_PATIENCE" \
  "RGB_W:$RGB_W" "RGB_H:$RGB_H" "TM_PORT:$TM_PORT" "N_NPCS:$N_NPCS"
do
  k="${pair%%:*}"; v="${pair#*:}"
  is_int "$v" || die "$k must be an integer (got '$v')"
done

if [[ "$RESTART_PER_TOWN" == "true" && -z "$CARLA_PATH" ]]; then
  die "-r requires -f PATH to CarlaUE4.sh / Carla.sh"
fi

# ---------- resolve towns ----------
declare -a MAPS=()
if [[ -z "$TOWNS" ]]; then
  MAPS=("${DEFAULT_MAPS[@]}")
else
  IFS=',' read -r -a MAPS <<< "$TOWNS"
  for i in "${!MAPS[@]}"; do MAPS[$i]="${MAPS[$i]//[[:space:]]/}"; done
fi
[[ "${#MAPS[@]}" -gt 0 ]] || die "No towns provided."

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

# ---------- flags ----------
redlight_flag=()
[[ "${NO_RED_LIGHT,,}" =~ ^(1|true|yes|on)$ ]] && redlight_flag+=(--no-red-light)

respawn_flag=()
[[ "$RESPAWN_ON_STALL" == "true" ]] && respawn_flag+=(--respawn-on-stall)

# Common args (as array; keep quoting safe)
common_args=(
  "--mode=$MODE"
  "--steps=$STEPS"
  "--fps=$FPS"
  "--target-speed=$TARGET_SPEED"
  "--min-speed-kmh=$MIN_SPEED_KMH"
  "--stall-patience=$STALL_PATIENCE"
  "--rgb-width" "$RGB_W"
  "--rgb-height" "$RGB_H"
  "--n-npcs=$N_NPCS"
  "--tm-port=$TM_PORT"
  "${redlight_flag[@]}"
  "${respawn_flag[@]}"
  "${EXTRA_ARGS[@]}"
)

# ---------- config echo ----------
info "Mode=$MODE  Steps=$STEPS  FPS=$FPS  Speed=$TARGET_SPEED km/h  NPCs=$N_NPCS  NoRed=$NO_RED_LIGHT"
info "MinSpeed=$MIN_SPEED_KMH  StallPatience=$STALL_PATIENCE  Size=${RGB_W}x${RGB_H}  TM_PORT=$TM_PORT"
info "RestartPerTown=$RESTART_PER_TOWN  OutRoot=$OUTPUT_DIR  CarlaPath=${CARLA_PATH:-<none>}"
info "Towns: $(IFS=,; echo "${MAPS[*]}")"
[[ "${#EXTRA_ARGS[@]}" -gt 0 ]] && info "Pass-through -> ${EXTRA_ARGS[*]}"

# ---------- execution ----------
if [[ "$RESTART_PER_TOWN" == "true" ]]; then
  for town in "${MAPS[@]}"; do
    run_dir="${town}-${TIMESTAMP}"
    info "Collecting on ${town} -> ${OUTPUT_DIR}/${run_dir}"
    run_cmd \
      --town="$town" \
      --out="${OUTPUT_DIR}/${run_dir}" \
      --restart-carla \
      --carla-path="$CARLA_PATH" \
      "${common_args[@]}"
  done
else
  joined="$(IFS=,; echo "${MAPS[*]}")"
  run_dir="all-${TIMESTAMP}"
  info "Collecting on towns: ${joined} -> ${OUTPUT_DIR}/${run_dir}"
  run_cmd \
    --towns="$joined" \
    --out="${OUTPUT_DIR}/${run_dir}" \
    "${common_args[@]}"
fi

info "Done."
