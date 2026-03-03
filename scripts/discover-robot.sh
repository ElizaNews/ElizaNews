#!/usr/bin/env bash
set -euo pipefail

# Discover AiNex robot on the local network
# Tries multiple methods: saved IP, mDNS, ARP scan, nmap

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/robot-config.sh"

CACHE_FILE="$SCRIPT_DIR/.robot-ip"
QUIET="${1:-}"

log() { [[ "$QUIET" == "-q" ]] || echo "$@"; }

# 1. Check if ROBOT_IP is already set
if [[ -n "$ROBOT_IP" ]]; then
    if ping -c1 -W1 "$ROBOT_IP" &>/dev/null; then
        log "Robot found at $ROBOT_IP (from env)"
        echo "$ROBOT_IP" > "$CACHE_FILE"
        echo "$ROBOT_IP"
        exit 0
    fi
    log "Warning: ROBOT_IP=$ROBOT_IP is not reachable"
fi

# 2. Check cached IP
if [[ -f "$CACHE_FILE" ]]; then
    CACHED_IP="$(cat "$CACHE_FILE")"
    if ping -c1 -W1 "$CACHED_IP" &>/dev/null; then
        log "Robot found at $CACHED_IP (cached)"
        echo "$CACHED_IP"
        exit 0
    fi
    log "Cached IP $CACHED_IP is stale, scanning..."
fi

# 3. Try mDNS (avahi)
if command -v avahi-resolve &>/dev/null; then
    log "Trying mDNS..."
    for name in ainex ubuntu ainex-robot; do
        IP=$(avahi-resolve -n "${name}.local" 2>/dev/null | awk '{print $2}' || true)
        if [[ -n "$IP" ]] && ping -c1 -W1 "$IP" &>/dev/null; then
            log "Robot found at $IP (mDNS: ${name}.local)"
            echo "$IP" > "$CACHE_FILE"
            echo "$IP"
            exit 0
        fi
    done
fi

# 4. ARP table scan — look for known Hiwonder MAC prefixes or SSH-capable hosts
log "Scanning network $SCAN_SUBNET for SSH hosts..."

# Quick ping sweep to populate ARP table
for i in $(seq 1 254); do
    BASE="${SCAN_SUBNET%.*}"
    ping -c1 -W0.1 "${BASE}.${i}" &>/dev/null &
done
wait 2>/dev/null

# Check ARP table for hosts and try SSH on each
CANDIDATES=()
while IFS= read -r ip; do
    [[ -z "$ip" ]] && continue
    CANDIDATES+=("$ip")
done < <(arp -n 2>/dev/null | grep -v incomplete | awk 'NR>1{print $1}' | sort -t. -k4 -n)

log "Found ${#CANDIDATES[@]} hosts, checking for AiNex..."

for ip in "${CANDIDATES[@]}"; do
    # Try SSH and check if it's the robot by looking for ros_ws
    if ssh $SSH_OPTS -o PasswordAuthentication=no -o BatchMode=yes \
        "${ROBOT_USER}@${ip}" "test -d /home/ubuntu/ros_ws" 2>/dev/null; then
        log "Robot found at $ip (SSH key auth)"
        echo "$ip" > "$CACHE_FILE"
        echo "$ip"
        exit 0
    fi
done

# 5. Try nmap if available
if command -v nmap &>/dev/null; then
    log "Running nmap scan for SSH on $SCAN_SUBNET..."
    while IFS= read -r ip; do
        [[ -z "$ip" ]] && continue
        log "  Trying $ip..."
        if ssh $SSH_OPTS -o PasswordAuthentication=no -o BatchMode=yes \
            "${ROBOT_USER}@${ip}" "test -d /home/ubuntu/ros_ws" 2>/dev/null; then
            log "Robot found at $ip (nmap + SSH)"
            echo "$ip" > "$CACHE_FILE"
            echo "$ip"
            exit 0
        fi
    done < <(nmap -sn "$SCAN_SUBNET" 2>/dev/null | grep "Nmap scan" | awk '{print $5}')
fi

echo ""
echo "ERROR: Could not find robot on network."
echo ""
echo "Options:"
echo "  1. Set IP manually:  export ROBOT_IP=192.168.x.x"
echo "  2. Set up SSH keys:  scripts/setup-ssh.sh <robot-ip>"
echo "  3. Check robot is powered on and connected to WiFi"
echo ""
exit 1
