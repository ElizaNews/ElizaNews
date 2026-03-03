#!/usr/bin/env bash
set -euo pipefail

# Connect to AiNex robot and start the bridge
# This runs the bridge on your dev machine, connecting to the robot's ROS stack
#
# Usage: scripts/connect.sh [--on-robot] [--verify]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/robot-config.sh"

ON_ROBOT=false
VERIFY=false
BACKEND="ros_real"

for arg in "$@"; do
    case "$arg" in
        --on-robot)  ON_ROBOT=true ;;
        --verify)    VERIFY=true ;;
        --mock)      BACKEND="mock" ;;
        --help|-h)
            echo "Usage: scripts/connect.sh [--on-robot] [--verify] [--mock]"
            echo ""
            echo "  --on-robot  Start bridge ON the robot (via SSH), not locally"
            echo "  --verify    Run E2E verification after connecting"
            echo "  --mock      Use mock backend (no robot needed)"
            echo ""
            echo "Default: start bridge locally, pointing at robot's ROS master"
            exit 0
            ;;
    esac
done

# Resolve robot IP (not needed for mock)
if [[ "$BACKEND" != "mock" ]]; then
    if [[ -n "$ROBOT_IP" ]]; then
        IP="$ROBOT_IP"
    elif [[ -f "$SCRIPT_DIR/.robot-ip" ]]; then
        IP="$(cat "$SCRIPT_DIR/.robot-ip")"
    else
        echo "Trying to discover robot..."
        IP="$("$SCRIPT_DIR/discover-robot.sh" -q)" || {
            echo "ERROR: Robot not found. Set ROBOT_IP or run: scripts/setup-ssh.sh <ip>"
            exit 1
        }
    fi
    echo "Robot IP: $IP"
fi

if $ON_ROBOT; then
    # --- Mode A: Run bridge ON the robot via SSH ---
    echo ""
    echo "=== Starting bridge ON robot ($IP) ==="
    echo "Bridge will listen on ws://${IP}:${BRIDGE_PORT}"
    echo "Press Ctrl+C to stop."
    echo ""

    ssh $SSH_OPTS "${ROBOT_USER}@${IP}" bash <<REMOTE
cd ${ROBOT_CODE_DIR} 2>/dev/null || cd /home/ubuntu
source /home/ubuntu/ros_ws/devel/setup.bash 2>/dev/null || true
export PYTHONPATH="${ROBOT_CODE_DIR}:\${PYTHONPATH:-}"
exec python3 -m bridge.server \
    --backend ros_real \
    --host 0.0.0.0 \
    --port ${BRIDGE_PORT} \
    --queue-size 256 \
    --max-commands-per-sec 30 \
    --deadman-timeout-sec 1.0
REMOTE

elif [[ "$BACKEND" == "mock" ]]; then
    # --- Mode B: Mock backend (no robot) ---
    echo ""
    echo "=== Starting mock bridge ==="
    echo "Bridge: ws://localhost:${BRIDGE_PORT}"
    echo "Press Ctrl+C to stop."
    echo ""

    cd "$HOST_CODE_DIR"
    export PYTHONPATH="$HOST_CODE_DIR:${PYTHONPATH:-}"
    exec python3 -m bridge.server \
        --backend mock \
        --host 0.0.0.0 \
        --port "$BRIDGE_PORT" \
        --queue-size 256 \
        --max-commands-per-sec 100 \
        --deadman-timeout-sec 60.0

else
    # --- Mode C: Run bridge locally, connect to robot's ROS ---
    echo ""
    echo "=== Starting bridge (local → robot) ==="
    echo "ROS Master: http://${IP}:11311"
    echo "Bridge:     ws://localhost:${BRIDGE_PORT}"
    echo "Camera:     http://${IP}:${CAMERA_PORT}/stream?topic=/camera/image_raw"
    echo "Press Ctrl+C to stop."
    echo ""

    # Check if robot's ROS is reachable
    if ! ssh $SSH_OPTS "${ROBOT_USER}@${IP}" "systemctl is-active --quiet start_app_node" 2>/dev/null; then
        echo "Warning: ROS stack doesn't appear to be running on robot."
        echo "Start it with: scripts/robot.sh start"
        echo ""
    fi

    cd "$HOST_CODE_DIR"
    export ROS_MASTER_URI="http://${IP}:11311"
    export ROS_IP="$(hostname -I | awk '{print $1}')"
    export PYTHONPATH="$HOST_CODE_DIR:${PYTHONPATH:-}"
    export AINEX_CAMERA_URL="http://${IP}:${CAMERA_PORT}/stream?topic=/camera/image_raw"

    if $VERIFY; then
        # Start bridge in background, run verification, then foreground
        python3 -m bridge.server \
            --backend "$BACKEND" \
            --host 0.0.0.0 \
            --port "$BRIDGE_PORT" \
            --queue-size 256 \
            --max-commands-per-sec 30 \
            --deadman-timeout-sec 1.0 &
        BRIDGE_PID=$!

        sleep 2
        echo ""
        echo "--- Running E2E verification ---"
        python3 bridge/scripts/verify_e2e.py --url "ws://localhost:${BRIDGE_PORT}" || true
        echo "--- Verification done ---"
        echo ""

        wait $BRIDGE_PID
    else
        exec python3 -m bridge.server \
            --backend "$BACKEND" \
            --host 0.0.0.0 \
            --port "$BRIDGE_PORT" \
            --queue-size 256 \
            --max-commands-per-sec 30 \
            --deadman-timeout-sec 1.0
    fi
fi
