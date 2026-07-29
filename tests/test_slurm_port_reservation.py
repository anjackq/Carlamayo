import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPOSITORY_ROOT / "scripts" / "slurm_port_reservation.sh"


def test_port_reservation_uses_lock_to_avoid_same_node_collision():
    script = f"""
set -euo pipefail
source {HELPER}
carlamayo_reserve_port_slot "" "$$"
first="$CARLAMAYO_RESERVED_PORT"
second=$(bash -c '
    source {HELPER}
    carlamayo_reserve_port_slot "" "$PPID"
    echo "$CARLAMAYO_RESERVED_PORT"
    carlamayo_release_port_slot
')
[[ "$first" != "$second" ]]
if bash -c "
    source {HELPER}
    carlamayo_reserve_port_slot $first 0
" >/dev/null 2>&1; then
    exit 1
fi
carlamayo_release_port_slot
"""

    subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
