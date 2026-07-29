#!/usr/bin/env bash
# Node-local four-port reservation helpers for CARLA Slurm wrappers.

# This file is sourced by the wrappers.  The open flock descriptor is retained
# by the calling shell until carlamayo_release_port_slot or process exit.

carlamayo_port_is_open() {
    local port="$1"
    (exec 9<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
}

carlamayo_try_reserve_port_slot() {
    local candidate="$1"
    local lock_fd
    local port

    exec {lock_fd}>"/tmp/carlamayo-port-${candidate}.lock"
    if ! flock -n "$lock_fd"; then
        exec {lock_fd}>&-
        return 1
    fi
    for ((port = candidate; port <= candidate + 3; port++)); do
        if carlamayo_port_is_open "$port"; then
            flock -u "$lock_fd"
            exec {lock_fd}>&-
            return 1
        fi
    done
    CARLAMAYO_RESERVED_PORT="$candidate"
    CARLAMAYO_PORT_LOCK_FD="$lock_fd"
    return 0
}

carlamayo_reserve_port_slot() {
    local requested_port="${1:-}"
    local job_id="${2:-0}"
    local base_index
    local attempt
    local candidate

    if ! command -v flock >/dev/null 2>&1; then
        echo "flock is required for CARLA port reservation." >&2
        return 1
    fi
    if [[ -n "$requested_port" ]]; then
        if ! [[ "$requested_port" =~ ^[0-9]+$ ]] \
            || ((requested_port < 1024 || requested_port > 65532)); then
            echo "Invalid explicit CARLAMAYO_CARLA_PORT: ${requested_port}" >&2
            return 1
        fi
        if ! carlamayo_try_reserve_port_slot "$requested_port"; then
            echo \
                "Explicit CARLA port slot ${requested_port}..$((requested_port + 3))" \
                "is busy on $(hostname)." \
                >&2
            return 1
        fi
        return 0
    fi

    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
        base_index=$((job_id % 10000))
    else
        base_index=0
    fi
    for ((attempt = 0; attempt < 10000; attempt++)); do
        candidate=$((20000 + 4 * ((base_index + attempt) % 10000)))
        if carlamayo_try_reserve_port_slot "$candidate"; then
            return 0
        fi
    done
    echo "No free four-port CARLA slot found on $(hostname)." >&2
    return 1
}

carlamayo_release_port_slot() {
    if [[ "${CARLAMAYO_PORT_LOCK_FD:-}" =~ ^[0-9]+$ ]]; then
        flock -u "$CARLAMAYO_PORT_LOCK_FD" 2>/dev/null || true
        exec {CARLAMAYO_PORT_LOCK_FD}>&-
    fi
}
