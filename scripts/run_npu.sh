#!/usr/bin/env bash
set -euo pipefail
cards="$1"
shift
source /usr/local/Ascend/cann/set_env.sh
status="$(npu-smi info)"
printf '%s\n' "$status"
NPU_STATUS="$status" python - "$cards" <<'PY'
import os
import re
import sys
status = os.environ['NPU_STATUS']
for card in map(int, sys.argv[1].split(',')):
    health = re.search(rf'^\|\s*{card}\s+910B2C\s*\|\s*(\w+)', status, re.M)
    idle = re.search(rf'No running processes found in NPU {card}\s*\|', status)
    if health is None or health[1] != 'OK' or idle is None:
        raise SystemExit(f'NPU {card} is not healthy and idle; command not started')
PY
export ASCEND_RT_VISIBLE_DEVICES="$cards"
exec "$@"
